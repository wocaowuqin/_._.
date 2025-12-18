import torch
import torch.nn as nn
import torch.nn.functional as F  # 🚨 补上这行 import
from torch_geometric.nn import GATv2Conv, global_mean_pool


class SharedEncoder(nn.Module):
    """
    [核心重构] 纯粹的图特征提取器 (Phase 2 & 3 共享)
    功能: (x, edge_index, req) -> graph_embedding
    """

    def __init__(self, node_feat_dim, edge_feat_dim, request_dim, hidden_dim, num_layers=3, heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. 节点特征嵌入
        self.node_lin = nn.Linear(node_feat_dim, hidden_dim)

        # 2. GAT 层 (标准的 GATv2)
        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_feat_dim, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_feat_dim, add_self_loops=False))

        # 3. 请求调制模块 (Request Modulator)
        self.req_modulator = nn.Sequential(
            nn.Linear(request_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, req_vec, batch):
        # --- 1. 维度修正 ---
        if req_vec.dim() == 1:
            req_vec = req_vec.unsqueeze(0)

        # --- 2. 节点嵌入 ---
        x = self.node_lin(x)

        # --- 3. GAT 消息传递 ---
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            # 使用 F.relu 或 torch.relu 均可
            x = F.relu(x)

            # --- 4. 请求调制 (Request Modulation) ---
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        req_expanded = req_vec[batch]
        gate = self.req_modulator(req_expanded)
        x = x * gate

        # --- 5. 图级池化 (Graph Embedding) ---
        graph_emb = global_mean_pool(x, batch)

        return graph_emb