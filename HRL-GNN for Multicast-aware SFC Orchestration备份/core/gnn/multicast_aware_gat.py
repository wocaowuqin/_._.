import torch
import torch.nn as nn
from core.gnn.shared_encoder import SharedEncoder


# multicast_aware_gat.py
class MulticastAwareGAT(nn.Module):
    def __init__(self,
                 node_feat_dim,
                 edge_feat_dim,
                 request_dim,
                 hidden_dim,
                 action_dim=None,
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

        # 2. 模式标志
        self.mode = "phase2" if action_dim is not None else "phase3"

        # 3. Phase 2 专用：输出层
        if self.mode == "phase2":
            # 🔥 关键修复：输出维度是 action_dim，不是 hidden_dim
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_dim)  # 直接输出 action_dim
            )
        else:
            self.output_layer = None

    def forward(self, x, edge_index, edge_attr, req_vec, batch, **kwargs):
        # 1. 编码
        z = self.encoder(x, edge_index, edge_attr, req_vec, batch)

        # 2. Phase 2: 输出 logits
        if self.mode == "phase2":
            return self.output_layer(z)

        # 3. Phase 3: 返回特征
        return z