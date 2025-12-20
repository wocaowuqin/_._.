# core/gnn/feature_builder.py
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import logging

logger = logging.getLogger(__name__)


class GNNFeatureBuilder:
    """
    GNN 特征处理器
    负责将 Environment 产出的单个状态数据，转换为模型可接受的 Batch 数据。
    """

    def __init__(self, device):
        self.device = device

    def to_pyg_data(self, state):
        """
        将 Env 返回的 Tuple 状态转换为 PyG 的 Data 对象
        State 格式: (x, edge_index, edge_attr, req_vec)
        """
        # 兼容性处理：如果 state 已经是 PyG Data，直接返回
        if hasattr(state, 'edge_index') and hasattr(state, 'x'):
            return state

        x, edge_index, edge_attr, req_vec = state

        # 确保是 Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        if not isinstance(req_vec, torch.Tensor):
            req_vec = torch.tensor(req_vec, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.req_vec = req_vec.unsqueeze(0)  # [1, D]

        return data

    def collate_fn(self, batch_data):
        """
        核心功能：将多个 transition 样本打包成 Batch
        支持输入为 dict (Agent存的) 或 tuple
        """
        batch_states = []
        batch_actions_high = []
        batch_actions_low = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_masks = []

        for item in batch_data:
            # 🔥 修复：支持字典类型的 Transition
            if isinstance(item, dict):
                state = item['state']
                action = item['action']
                reward = item['reward']
                next_state = item['next_state']
                done = item['done']
                masks = item.get('next_valid_mask')  # 可能为 None
            else:
                # 旧的 Tuple 解包方式 (保留兼容性)
                try:
                    state, action, reward, next_state, done, masks = item
                except ValueError:
                    # 如果 tuple 长度不对，尝试忽略 masks
                    state, action, reward, next_state, done = item[:5]
                    masks = None

            # 1. 处理状态
            batch_states.append(self.to_pyg_data(state))
            batch_next_states.append(self.to_pyg_data(next_state))

            # 2. 处理动作
            if isinstance(action, (tuple, list, np.ndarray)):
                high_act, low_act = action
            else:
                high_act, low_act = 0, action  # fallback

            batch_actions_high.append(high_act)
            batch_actions_low.append(low_act)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_masks.append(masks)

        # === Batch 拼接 ===
        batched_state = Batch.from_data_list(batch_states).to(self.device)
        batched_next_state = Batch.from_data_list(batch_next_states).to(self.device)

        # 拼接全局特征 req_vec [B, D]
        if hasattr(batch_states[0], 'req_vec'):
            batched_state.req_vec = torch.cat([d.req_vec for d in batch_states], dim=0).to(self.device)

        if hasattr(batch_next_states[0], 'req_vec'):
            batched_next_state.req_vec = torch.cat([d.req_vec for d in batch_next_states], dim=0).to(self.device)

        # 转换为 Tensor
        actions_high = torch.tensor(batch_actions_high, dtype=torch.long, device=self.device)
        actions_low = torch.tensor(batch_actions_low, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Mask 处理
        # 如果 batch_masks 中有 None，我们需要构造全 1 Mask 防止报错
        if any(m is None for m in batch_masks):
            # 构造默认 Mask (假设 mask 全为 1)
            # 注意：这里无法动态获取 dim，只能尽力防御
            masks_high = None
            masks_low = None
        else:
            masks_high = torch.tensor(np.array([m[0] for m in batch_masks]), dtype=torch.float32, device=self.device)
            masks_low = torch.tensor(np.array([m[1] for m in batch_masks]), dtype=torch.float32, device=self.device)

        return {
            'state': batched_state,
            'next_state': batched_next_state,
            'action': (actions_high, actions_low),
            'reward': rewards,
            'done': dones,
            'mask': (masks_high, masks_low)  # 可能包含 None
        }