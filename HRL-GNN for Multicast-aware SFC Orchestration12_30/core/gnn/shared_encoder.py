"""
core/gnn/shared_encoder.py
GNN 共享编码器 - 终极修复版 (带自动维度适配和详细日志)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)

class SharedEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SharedEncoder, self).__init__()

        # ====================================================
        # 1. 万能参数解析 (兼容 Config对象 / 字典 / 位置参数)
        # ====================================================

        # 默认配置
        node_feat_dim = 17
        edge_feat_dim = 5
        request_dim = 24
        hidden_dim = 128
        self.num_layers = kwargs.get('num_layers', 2)

        # 情况 A: 传入了一个 config 对象或字典
        if len(args) == 1 and (isinstance(args[0], dict) or hasattr(args[0], 'get') or hasattr(args[0], 'gnn')):
            cfg = args[0]
            def get_cfg(key, default):
                if isinstance(cfg, dict): return cfg.get(key, default)
                val = getattr(cfg, key, None)
                if val is not None: return val
                if hasattr(cfg, 'gnn'):
                    gnn = getattr(cfg, 'gnn')
                    return gnn.get(key, default) if isinstance(gnn, dict) else getattr(gnn, key, default)
                return default

            node_feat_dim = get_cfg('node_feat_dim', node_feat_dim)
            edge_feat_dim = get_cfg('edge_feat_dim', edge_feat_dim)
            request_dim = get_cfg('request_feat_dim', request_dim)
            hidden_dim = get_cfg('hidden_dim', hidden_dim)

        # 情况 B: 位置参数
        elif len(args) >= 3:
            node_feat_dim = args[0]
            edge_feat_dim = args[1]
            request_dim = args[2]
            if len(args) > 3: hidden_dim = args[3]

        # 情况 C: 关键字参数覆盖
        node_feat_dim = kwargs.get('node_feat_dim', node_feat_dim)
        edge_feat_dim = kwargs.get('edge_feat_dim', edge_feat_dim)
        request_dim = kwargs.get('request_feat_dim', request_dim)
        hidden_dim = kwargs.get('hidden_dim', hidden_dim)

        # 保存期望维度
        self.node_feat_dim = int(node_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.request_dim = int(request_dim)
        self.hidden_dim = int(hidden_dim)

        logger.info(f"🔍 [SharedEncoder] Init: Node={self.node_feat_dim}, Edge={self.edge_feat_dim}, Req={self.request_dim}, Hidden={self.hidden_dim}")

        # ====================================================
        # 2. 网络构建
        # ====================================================

        # 第一层 GAT
        self.conv1 = GATv2Conv(
            in_channels=self.node_feat_dim,
            out_channels=self.hidden_dim,
            heads=4,
            edge_dim=self.edge_feat_dim if self.edge_feat_dim > 0 else None,
            concat=False
        )

        # 第二层 GAT
        self.conv2 = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=4,
            edge_dim=self.edge_feat_dim if self.edge_feat_dim > 0 else None,
            concat=False
        )

        # 请求特征层
        if self.request_dim > 0:
            self.req_fc = nn.Linear(self.request_dim, self.hidden_dim)
        else:
            self.req_fc = None

        # 融合层
        self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_dim = self.hidden_dim

        # 状态标记
        self._warned = set()

    def _fix_dim(self, tensor, expected_dim, name="tensor"):
        """自动修复维度不匹配"""
        if tensor is None: return None

        # 获取最后一个维度
        actual_dim = tensor.shape[-1]

        if actual_dim == expected_dim:
            return tensor

        if name not in self._warned:
            logger.warning(f"⚠️ [SharedEncoder] {name} dim mismatch! Expected {expected_dim}, got {actual_dim}. Auto-fixing...")
            self._warned.add(name)

        if actual_dim < expected_dim:
            # 填充
            padding_shape = list(tensor.shape)
            padding_shape[-1] = expected_dim - actual_dim
            padding = torch.zeros(padding_shape, device=tensor.device)
            return torch.cat([tensor, padding], dim=-1)
        else:
            # 截断
            return tensor[..., :expected_dim]

    def forward(self, x, edge_index, edge_attr=None, req_vec=None, batch=None):
        device = x.device

        # 1. 自动修复节点特征
        x = self._fix_dim(x, self.node_feat_dim, "node_feat")

        # 2. 自动修复边缘特征
        if self.edge_feat_dim > 0:
            if edge_attr is None:
                num_edges = edge_index.shape[1]
                edge_attr = torch.zeros(num_edges, self.edge_feat_dim, device=device)
            else:
                edge_attr = self._fix_dim(edge_attr, self.edge_feat_dim, "edge_attr")

        # 3. GAT 卷积
        try:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        except RuntimeError as e:
            logger.error(f"❌ [SharedEncoder] GAT Forward Failed: {e}")
            logger.error(f"   x shape: {x.shape}")
            logger.error(f"   edge_attr shape: {edge_attr.shape if edge_attr is not None else 'None'}")
            raise e

        # 4. 请求特征处理
        if self.req_fc is not None:
            if req_vec is None:
                batch_size = 1 if batch is None else (batch.max().item() + 1)
                req_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
            else:
                if req_vec.dim() == 1: req_vec = req_vec.unsqueeze(0)
                req_vec = self._fix_dim(req_vec, self.request_dim, "req_vec")
                req_emb = self.req_fc(req_vec)
        else:
             batch_size = 1 if batch is None else (batch.max().item() + 1)
             req_emb = torch.zeros(batch_size, self.hidden_dim, device=device)

        # 5. 智能扩展
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        if req_emb.dim() == 1:
            req_expanded = req_emb.unsqueeze(0).expand(x.size(0), -1)
        else:
            max_batch_idx = batch.max().item()
            if req_emb.size(0) <= max_batch_idx:
                req_expanded = req_emb[0].unsqueeze(0).expand(x.size(0), -1)
            else:
                req_expanded = req_emb[batch]

        # 6. 融合
        combined = torch.cat([x, req_expanded], dim=-1)
        out = self.fusion(combined)

        return out