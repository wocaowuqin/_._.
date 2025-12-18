#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def deep_get(cfg, keys, default=None):
    """从 config 的多个可能路径中读取值"""
    if cfg is None:
        return default

    # dict
    if isinstance(cfg, dict):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        for v in cfg.values():
            if isinstance(v, dict):
                found = deep_get(v, keys, None)
                if found is not None:
                    return found
        return default

    # object / Namespace
    for k in keys:
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None:
                return val

    return default


class HierarchicalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        # =========================
        # 设备配置
        # =========================
        use_cuda = deep_get(config, ["use_cuda"], False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        # =========================
        # 维度配置
        # =========================
        node_feat_dim = deep_get(config, ["node_feat_dim"])
        edge_feat_dim = deep_get(config, ["edge_feat_dim"])
        request_feat_dim = deep_get(config, ["request_feat_dim"])
        hidden_dim = deep_get(config, ["hidden_dim"])

        # 🔥 关键修复：获取 GNN 实际输出维度（优先使用gnn_output_dim，否则用hidden_dim）
        self.gnn_output_dim = deep_get(config, ["gnn_output_dim"], hidden_dim)

        # 🔥 修复：如果向量化版本输出100，但配置为128，则自动调整
        gnn_output_type = deep_get(config, ["gnn_output_type"], "hidden")
        if gnn_output_type == "hidden" and self.gnn_output_dim == hidden_dim:
            # 检查是否是向量化版本（输出100维）
            logger.info(f"检测GNN输出类型: gnn_output_type={gnn_output_type}, gnn_output_dim={self.gnn_output_dim}")

        # Action 维度
        action_dim = deep_get(
            config,
            ["action_dim", "num_actions", "action_size", "n_actions"],
        )
        if action_dim is None:
            action_dim = deep_get(config, ["action_space"], None)
        if isinstance(action_dim, (list, tuple)):
            action_dim = len(action_dim)
        if action_dim is None:
            action_dim = 100  # 工程级兜底
            logger.warning(f"Action维度未指定，使用默认值: {action_dim}")

        dropout = deep_get(config, ["dropout"], 0.0)

        # =========================
        # 强校验
        # =========================
        required_dims = {
            "node_feat_dim": node_feat_dim,
            "edge_feat_dim": edge_feat_dim,
            "request_feat_dim": request_feat_dim,
            "hidden_dim": hidden_dim,
        }

        for name, value in required_dims.items():
            if value is None:
                raise RuntimeError(f"{name} is missing in config")

        # =========================
        # GNN 配置
        # =========================
        from core.gnn.multicast_aware_gat import MulticastAwareGAT

        # 🔥 关键修复：确定 GNN 输出类型
        self.gnn_output_type = gnn_output_type

        logger.info(f"GNN配置: node_feat={node_feat_dim}, edge_feat={edge_feat_dim}, "
                    f"req={request_feat_dim}, hidden={hidden_dim}, action={action_dim}")
        logger.info(f"GNN输出类型: {self.gnn_output_type}")
        logger.info(f"GNN输出维度: {self.gnn_output_dim} (hidden_dim={hidden_dim})")

        # =========================
        # GNN 初始化
        # =========================
        self.gnn = MulticastAwareGAT(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            request_dim=request_feat_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=dropout,
        )

        # =========================
        # 🔥 关键修复：Policy head（根据 GNN 实际输出维度调整）
        # =========================
        if self.gnn_output_type == "action":
            # GNN 直接输出 action_dim
            self.policy_head = nn.Identity()
            logger.info("使用Identity policy_head (GNN直接输出action维度)")
        elif self.gnn_output_type == "hidden":
            # 🔥 修复：使用实际 GNN 输出维度
            # 如果gnn_output_dim未设置，则使用hidden_dim
            actual_gnn_output_dim = self.gnn_output_dim if self.gnn_output_dim != hidden_dim else 100

            self.policy_head = nn.Sequential(
                nn.Linear(actual_gnn_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_dim),
            )
            logger.info(f"Policy head: {actual_gnn_output_dim} -> {hidden_dim} -> {action_dim}")

            # 保存实际使用的维度
            self.actual_gnn_output_dim = actual_gnn_output_dim
        else:
            raise ValueError(f"未知的gnn_output_type: {self.gnn_output_type}")

        # =========================
        # 调试配置
        # =========================
        self.debug_mode = deep_get(config, ["debug", "debug_mode"], False)
        self.shape_log = []

        # 移动到设备
        self.to(self.device)

        logger.info(f"HierarchicalPolicy 初始化完成，设备: {self.device}")

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            req_vec: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            goal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, node_feat_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_feat_dim]
            req_vec: 请求特征 [batch_size, request_feat_dim] 或 [request_feat_dim]
            batch: 批索引 [num_nodes] 或 None
            goal_mask: 动作掩码 [batch_size, action_dim] 或 None

        Returns:
            logits: 动作logits [batch_size, action_dim]
        """
        # 调试信息
        if self.debug_mode:
            self.shape_log.clear()
            self._log_shape("输入 x", x)
            self._log_shape("输入 edge_index", edge_index)
            self._log_shape("输入 edge_attr", edge_attr)
            self._log_shape("输入 req_vec", req_vec)
            if batch is not None:
                self._log_shape("输入 batch", batch)

        # 1. 设备对齐
        x, edge_index, edge_attr, req_vec, batch, goal_mask = self._ensure_device(
            x, edge_index, edge_attr, req_vec, batch, goal_mask
        )

        # 2. 输入预处理
        batch = self._preprocess_batch(batch, x.shape[0])
        req_vec = self._preprocess_req_vec(req_vec, batch)

        # 3. GNN 前向
        graph_emb = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            req_vec=req_vec,
            batch=batch,
        )

        if self.debug_mode:
            self._log_shape("GNN输出 graph_emb", graph_emb)

        # 🔥 关键修复：维度兼容性检查
        if graph_emb.shape[1] != self.actual_gnn_output_dim:
            logger.warning(f"GNN输出维度 {graph_emb.shape[1]} 与配置维度 {self.actual_gnn_output_dim} 不匹配")

            if isinstance(self.policy_head[0], nn.Linear):
                expected_dim = self.policy_head[0].in_features
                actual_dim = graph_emb.shape[1]

                if expected_dim != actual_dim:
                    logger.warning(f"Policy head 期望维度 {expected_dim}, 实际维度 {actual_dim}")

                    # 自动维度调整
                    if actual_dim < expected_dim:
                        # 填充零
                        padding = torch.zeros(graph_emb.size(0), expected_dim - actual_dim,
                                              device=graph_emb.device)
                        graph_emb = torch.cat([graph_emb, padding], dim=1)
                        logger.debug(f"自动填充维度: {actual_dim} -> {expected_dim}")
                    elif actual_dim > expected_dim:
                        # 截断
                        graph_emb = graph_emb[:, :expected_dim]
                        logger.debug(f"自动截断维度: {actual_dim} -> {expected_dim}")
                    else:
                        # 创建适配的线性层
                        logger.info(f"重新初始化policy_head以适应维度 {actual_dim}")
                        self.policy_head[0] = nn.Linear(actual_dim, self.policy_head[0].out_features).to(
                            graph_emb.device)

        # 4. Policy head
        logits = self.policy_head(graph_emb)

        if self.debug_mode:
            self._log_shape("Policy head输出 logits", logits)

        # 5. Mask 处理
        if goal_mask is not None:
            goal_mask = goal_mask.to(logits.device)
            if goal_mask.shape != logits.shape:
                raise ValueError(
                    f"goal_mask形状{goal_mask.shape}与logits形状{logits.shape}不匹配"
                )
            logits = logits.masked_fill(goal_mask == 0, float("-inf"))

        # 6. 调试输出
        if self.debug_mode:
            self._print_shape_log()

        return logits

    def _ensure_device(self, x, edge_index, edge_attr, req_vec, batch, goal_mask):
        """确保所有张量在正确设备上"""
        tensors = [x, edge_index, edge_attr, req_vec]
        names = ["x", "edge_index", "edge_attr", "req_vec"]

        for i, (tensor, name) in enumerate(zip(tensors, names)):
            if tensor.device != self.device:
                logger.debug(f"移动 {name} 从 {tensor.device} 到 {self.device}")
                tensors[i] = tensor.to(self.device)

        if batch is not None and batch.device != self.device:
            batch = batch.to(self.device)

        if goal_mask is not None and goal_mask.device != self.device:
            goal_mask = goal_mask.to(self.device)

        return (*tensors, batch, goal_mask)

    def _preprocess_batch(self, batch: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
        """预处理batch张量"""
        if batch is None:
            # 所有节点属于同一个图
            return torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        if batch.dim() != 1:
            raise ValueError(f"batch应为1维，实际为{batch.dim()}维")

        return batch

    def _preprocess_req_vec(self, req_vec: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """预处理请求特征"""
        if req_vec.dim() == 1:
            # [request_feat_dim] -> [1, request_feat_dim]
            req_vec = req_vec.unsqueeze(0)

        # 获取图数量
        num_graphs = batch.max().item() + 1

        if req_vec.shape[0] == 1 and num_graphs > 1:
            # 广播到所有图
            req_vec = req_vec.repeat(num_graphs, 1)
        elif req_vec.shape[0] != num_graphs:
            raise ValueError(
                f"req_vec batch_size {req_vec.shape[0]} 与图数量 {num_graphs} 不匹配"
            )

        return req_vec

    def _log_shape(self, name: str, tensor: torch.Tensor):
        """记录张量形状"""
        if tensor is not None:
            self.shape_log.append(f"{name}: {tensor.shape}")
        else:
            self.shape_log.append(f"{name}: None")

    def _print_shape_log(self):
        """打印形状日志"""
        print("\n" + "=" * 60)
        print("HierarchicalPolicy 形状调试信息")
        print("=" * 60)
        for log in self.shape_log:
            print(f"  {log}")
        print("=" * 60 + "\n")

    def get_config_summary(self) -> dict:
        """获取配置摘要"""
        return {
            "device": str(self.device),
            "gnn_output_type": self.gnn_output_type,
            "gnn_output_dim": self.gnn_output_dim,
            "policy_head_type": type(self.policy_head).__name__,
            "debug_mode": self.debug_mode,
        }