# core/hrl/high_policy.py

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool  # 显式导入池化函数

# 使用正确的相对导入或绝对导入
try:
    from core.gnn import MulticastAwareGAT
except ImportError:
    # 备用导入路径
    from ..gnn.multicast_aware_gat import MulticastAwareGAT

from .mid_policy import MidLevelPolicy
from .low_policy import LowLevelPolicy


class HierarchicalPolicy(nn.Module):
    """
    高层策略容器 (HRL Main Module)
    整合 GNN 特征提取、中层目标选择和低层动作选择
    """

    def __init__(self, config):
        super(HierarchicalPolicy, self).__init__()

        gnn_cfg = config['gnn']
        env_cfg = config['env']

        # 1. GNN 骨干网络 (特征提取)
        self.gnn = MulticastAwareGAT(
            node_feat_dim=gnn_cfg['node_feat_dim'],
            edge_feat_dim=gnn_cfg['edge_feat_dim'],
            request_dim=gnn_cfg['request_feat_dim'],
            hidden_dim=gnn_cfg['hidden_dim'],
            num_heads=gnn_cfg.get('num_heads', 4),
            num_gat_layers=gnn_cfg.get('num_gat_layers', 3)
        )

        # 2. 状态融合层 (Graph Embedding + Request Vector)
        # 显式转为 int 防止 YAML 读取类型警告
        self.fusion_dim = int(gnn_cfg['hidden_dim'] + gnn_cfg['request_feat_dim'])
        self.shared_hidden_dim = 256

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, self.shared_hidden_dim),
            nn.LayerNorm(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gnn_cfg.get('dropout', 0.1))
        )

        # 3. 实例化子策略
        self.mid_policy = MidLevelPolicy(
            input_dim=self.shared_hidden_dim,
            action_dim=env_cfg['nb_high_level_goals'],
            hidden_dim=128
        )

        self.low_policy = LowLevelPolicy(
            input_dim=self.shared_hidden_dim,
            action_dim=env_cfg['nb_low_level_actions'],
            hidden_dim=128
        )

    def forward(self, x, edge_index, edge_attr, req_vec, dest_indices=None, batch=None):
        """
        前向传播：从原始图数据到所有策略的 logits
        """
        # 1. GNN 特征提取
        # 注意：GNN 返回的是节点级嵌入
        node_emb, dest_set_feat, sharing_scores = self.gnn(
            x, edge_index, edge_attr, req_vec,
            dest_indices=dest_indices if dest_indices is not None else [],
            batch=batch
        )

        # 2. 图级聚合 (Pooling)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # [Num_Nodes, Hidden] -> [Batch_Size, Hidden]
        graph_emb = global_mean_pool(node_emb, batch)

        # 3. 融合请求向量
        if req_vec.dim() == 1:
            req_vec = req_vec.unsqueeze(0)  # [1, D]

        # 拼接: [B, Hidden] + [B, Req_Dim]
        combined = torch.cat([graph_emb, req_vec], dim=1)

        # 4. 得到共享的隐层状态
        state_emb = self.fusion_layer(combined)

        # 5. 子策略前向传播
        mid_logits, mid_value = self.mid_policy(state_emb)
        low_logits, low_value = self.low_policy(state_emb)

        return mid_logits, low_logits, mid_value, low_value

    def get_embedding(self, x, edge_index, edge_attr, req_vec, dest_indices=None, batch=None):
        """
        🟢 [修复] 辅助方法：仅获取状态嵌入
        修复了参数缺失、解包错误和缺少 Pooling 的问题
        """
        with torch.no_grad():
            # 1. 调用 GNN (必须传入 dest_indices)
            node_emb, _, _ = self.gnn(
                x, edge_index, edge_attr, req_vec,
                dest_indices=dest_indices if dest_indices is not None else [],
                batch=batch
            )

            # 2. 手动 Pooling (必须做这一步，否则维度对不上)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_emb = global_mean_pool(node_emb, batch)

            # 3. 拼接
            if req_vec.dim() == 1:
                req_vec = req_vec.unsqueeze(0)
            combined = torch.cat([graph_emb, req_vec], dim=1)

            return self.fusion_layer(combined)
