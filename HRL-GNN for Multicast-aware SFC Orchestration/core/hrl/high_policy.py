#高层选择节点
import torch
import torch.nn as nn
from core.gnn import MulticastAwareGAT
from core.hrl.mid_policy import MidLevelPolicy
from core.hrl.low_policy import LowLevelPolicy


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
            node_in_dim=gnn_cfg['node_feat_dim'],
            edge_in_dim=gnn_cfg['edge_feat_dim'],
            hidden_dim=gnn_cfg['hidden_dim'],
            num_heads=gnn_cfg.get('num_heads', 4),
            dropout=gnn_cfg.get('dropout', 0.1)
        )

        # 2. 状态融合层 (Graph Embedding + Request Vector)
        # 融合维度 = GNN Hidden + Request Dim
        self.fusion_dim = gnn_cfg['hidden_dim'] + gnn_cfg['request_feat_dim']
        self.shared_hidden_dim = 256

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, self.shared_hidden_dim),
            nn.LayerNorm(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 3. 实例化子策略
        # Mid Policy (Goal Selection)
        self.mid_policy = MidLevelPolicy(
            input_dim=self.shared_hidden_dim,
            action_dim=env_cfg['nb_high_level_goals'],
            hidden_dim=128
        )

        # Low Policy (Action Selection)
        self.low_policy = LowLevelPolicy(
            input_dim=self.shared_hidden_dim,
            action_dim=env_cfg['nb_low_level_actions'],
            hidden_dim=128
        )

    def forward(self, x, edge_index, edge_attr, req_vec, batch=None):
        """
        前向传播：从原始图数据到所有策略的 logits

        Returns:
            mid_logits: [B, Num_Goals]
            low_logits: [B, Num_Actions]
            mid_value:  [B, 1]
            low_value:  [B, 1]
        """
        # 1. GNN 特征提取
        # node_emb: [N, H], graph_emb: [B, H]
        node_emb, graph_emb = self.gnn(x, edge_index, edge_attr, batch)

        # 2. 融合请求向量
        if req_vec.dim() == 1:
            req_vec = req_vec.unsqueeze(0)  # [1, D]

        # 拼接: [B, Hidden] + [B, Req_Dim] -> [B, Fusion_Dim]
        combined = torch.cat([graph_emb, req_vec], dim=1)

        # 得到共享的隐层状态
        state_emb = self.fusion_layer(combined)

        # 3. 子策略前向传播
        mid_logits, mid_value = self.mid_policy(state_emb)
        low_logits, low_value = self.low_policy(state_emb)

        return mid_logits, low_logits, mid_value, low_value

    def get_embedding(self, x, edge_index, edge_attr, req_vec, batch=None):
        """辅助方法：仅获取状态嵌入 (用于可视化或调试)"""
        with torch.no_grad():
            _, graph_emb = self.gnn(x, edge_index, edge_attr, batch)
            if req_vec.dim() == 1: req_vec = req_vec.unsqueeze(0)
            combined = torch.cat([graph_emb, req_vec], dim=1)
            return self.fusion_layer(combined)