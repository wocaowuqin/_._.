#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multicast_aware_gat.py
请求感知的多播图注意力网络

创新点:
1. 多目标集合编码 (Set Transformer)
2. 请求调制注意力
3. VNF共享潜力预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import List, Optional, Tuple


class SetTransformer(nn.Module):
    """
    多目标集合编码器 (排列不变性)

    理论基础: Deep Sets (Zaheer et al., NeurIPS 2017)
    f({x1, ..., xn}) = ρ(Σ φ(xi))
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # φ: 元素级编码
        self.element_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Multi-head Self-Attention (捕获目标间关系)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # ρ: 聚合函数
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dest_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dest_features: [num_dests, input_dim] 目标节点特征
        Returns:
            aggregated: [hidden_dim] 聚合后的集合表示
        """
        # 1. 元素级编码
        encoded = self.element_encoder(dest_features)  # [num_dests, hidden_dim]

        # 2. Self-Attention (捕获目标间相关性)
        # 例如: 两个目标如果距离很近,可能共享VNF
        attn_out, _ = self.self_attn(
            encoded.unsqueeze(0),  # [1, num_dests, hidden_dim]
            encoded.unsqueeze(0),
            encoded.unsqueeze(0)
        )
        attn_out = attn_out.squeeze(0)  # [num_dests, hidden_dim]

        # 3. 排列不变聚合 (求和)
        aggregated = torch.sum(attn_out, dim=0)  # [hidden_dim]

        # 4. 最终映射
        output = self.aggregator(aggregated)

        return output


class RequestModulatedAttention(nn.Module):
    """
    请求调制的图注意力

    创新点: 注意力权重由请求特征动态调制
    α_ij = attention(h_i, h_j, request_vec)
    """

    def __init__(self, node_dim: int, request_dim: int, hidden_dim: int):
        super().__init__()

        # 标准GAT注意力计算
        self.attn_src = nn.Linear(node_dim, hidden_dim)
        self.attn_dst = nn.Linear(node_dim, hidden_dim)

        # 🔥 创新: 请求调制网络
        self.request_modulator = nn.Sequential(
            nn.Linear(request_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重计算
        self.attn_weight = nn.Linear(hidden_dim * 3, 1)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor,
                request_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_i: [batch, node_dim] 源节点特征
            h_j: [batch, node_dim] 目标节点特征
            request_vec: [batch, request_dim] 请求向量
        Returns:
            alpha: [batch, 1] 注意力权重
        """
        # 1. 节点特征映射
        src_feat = self.attn_src(h_i)  # [batch, hidden_dim]
        dst_feat = self.attn_dst(h_j)  # [batch, hidden_dim]

        # 2. 🔥 请求调制
        req_feat = self.request_modulator(request_vec)  # [batch, hidden_dim]

        # 3. 拼接并计算权重
        combined = torch.cat([src_feat, dst_feat, req_feat], dim=-1)
        alpha = self.attn_weight(combined)  # [batch, 1]

        return torch.sigmoid(alpha)


class VNFSharingPredictor(nn.Module):
    """
    VNF共享潜力预测器

    功能: 预测每个节点作为VNF共享点的潜力
    输入: 节点特征 + 目标集合特征
    输出: 共享潜力分数 [0, 1]
    """

    def __init__(self, node_dim: int, dest_set_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(node_dim + dest_set_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, node_feat: torch.Tensor,
                dest_set_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: [num_nodes, node_dim]
            dest_set_feat: [dest_set_dim] 目标集合特征
        Returns:
            scores: [num_nodes] 每个节点的共享潜力
        """
        # 扩展dest_set_feat以匹配节点数
        dest_expanded = dest_set_feat.unsqueeze(0).expand(
            node_feat.size(0), -1
        )  # [num_nodes, dest_set_dim]

        # 拼接
        combined = torch.cat([node_feat, dest_expanded], dim=-1)

        # 预测
        scores = self.predictor(combined).squeeze(-1)  # [num_nodes]

        return scores


class MulticastAwareGAT(nn.Module):
    """
    完整的多播感知GAT网络

    整合所有创新组件
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 request_dim: int, hidden_dim: int = 128,
                 num_gat_layers: int = 3, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ===== 组件1: 基础特征编码 =====
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feat_dim, hidden_dim)

        # ===== 组件2: 多目标集合编码器 (创新) =====
        self.dest_set_encoder = SetTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # ===== 组件3: 请求调制GAT层 (创新) =====
        self.gat_layers = nn.ModuleList()
        self.request_modulators = nn.ModuleList()

        for _ in range(num_gat_layers):
            # 标准GAT层
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )

            # 请求调制器 (每层一个)
            self.request_modulators.append(
                RequestModulatedAttention(
                    node_dim=hidden_dim,
                    request_dim=request_dim,
                    hidden_dim=hidden_dim
                )
            )

        # Layer Norm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])

        # ===== 组件4: VNF共享潜力预测 (创新) =====
        self.sharing_predictor = VNFSharingPredictor(
            node_dim=hidden_dim,
            dest_set_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, request_vec: torch.Tensor,
                dest_indices: List[int],
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播

        Args:
            x: [num_nodes, node_feat_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_feat_dim] 边特征
            request_vec: [request_dim] 请求向量
            dest_indices: List[int] 目标节点索引
            batch: [num_nodes] batch索引 (可选)

        Returns:
            node_embeddings: [num_nodes, hidden_dim] 节点嵌入
            dest_set_embedding: [hidden_dim] 目标集合嵌入
            sharing_scores: [num_nodes] VNF共享潜力分数
        """
        device = x.device

        # ===== Step 1: 初始特征编码 =====
        x = self.node_embedding(x)  # [num_nodes, hidden_dim]
        e = self.edge_embedding(edge_attr)  # [num_edges, hidden_dim]

        # ===== Step 2: 目标集合编码 (创新) =====
        dest_features = x[dest_indices]  # [num_dests, hidden_dim]
        dest_set_feat = self.dest_set_encoder(dest_features)  # [hidden_dim]

        # ===== Step 3: 请求调制的GAT传播 (创新) =====
        for gat_layer, modulator, norm in zip(
                self.gat_layers, self.request_modulators, self.layer_norms
        ):
            residual = x

            # 原始代码（第 275-285 行）
            # 标准GAT传播
            x_gat = gat_layer(x, edge_index, e)  # [num_nodes, hidden_dim]

            # ===== 正确的请求调制（支持 Batch）=====
            # request_vec: [B, request_dim]
            assert request_vec.dim() == 2, \
                f"request_vec must be [B, D], got {request_vec.shape}"

            if batch is None:
                # 单图 / Phase1 情况
                request_expanded = request_vec[0].unsqueeze(0).expand(x.size(0), -1)
            else:
                # Phase2 Batch：按 batch 索引对齐到每个节点
                request_expanded = request_vec[batch]  # [num_nodes, request_dim]

            modulation_weights = modulator(
                x, x, request_expanded
            )  # [num_nodes, 1]

            modulation_weights = modulator(
                x, x, request_expanded
            )  # [num_nodes, 1]


            # 应用调制
            x_modulated = x_gat * modulation_weights

            # 残差连接 + LayerNorm
            x = norm(residual + x_modulated)

        # ===== Step 4: VNF共享潜力预测 (创新) =====
        sharing_scores = self.sharing_predictor(x, dest_set_feat)

        # ===== Step 5: 图级聚合 =====
        if batch is None:
            graph_emb = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=device))
        else:
            graph_emb = global_mean_pool(x, batch)

        return x, dest_set_feat, sharing_scores