import torch
import torch.nn as nn
from core.gnn.shared_encoder import SharedEncoder


class MulticastAwareGAT(nn.Module):
    """
    修复后的 MulticastAwareGAT

    功能：纯粹的特征提取，不包含 output_layer
    输出：hidden_dim 维度的图嵌入
    """

    def __init__(self,
                 node_feat_dim,
                 edge_feat_dim,
                 request_dim,
                 hidden_dim,
                 action_dim=None,  # 保留参数用于兼容性，但不使用
                 num_layers=3,
                 num_heads=4,
                 dropout=0.0,
                 **kwargs):
        super().__init__()

        # 1. 实例化共享 Encoder
        self.encoder = SharedEncoder(
            node_feat_dim,
            edge_feat_dim,
            request_dim,
            hidden_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        # 2. 保存配置
        self.hidden_dim = hidden_dim
        self.mode = "phase2" if action_dim is not None else "phase3"

        # 3. 🔥 关键修复：移除 output_layer
        # GNN 只负责特征提取，始终输出 hidden_dim
        # Policy head 由 high_policy.py 处理
        self.output_layer = None

    def forward(self, x, edge_index, edge_attr, req_vec, batch, **kwargs):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, node_feat_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_feat_dim]
            req_vec: 请求特征 [batch_size, request_dim] 或 [request_dim]
            batch: 批次索引 [num_nodes] 或 None

        Returns:
            graph_emb: 图嵌入 [batch_size, hidden_dim]
        """
        # 编码（获取图嵌入）
        z = self.encoder(x, edge_index, edge_attr, req_vec, batch)

        # 🔥 关键修复：始终返回特征（hidden_dim）
        # 不在 GNN 内部输出 action logits
        return z