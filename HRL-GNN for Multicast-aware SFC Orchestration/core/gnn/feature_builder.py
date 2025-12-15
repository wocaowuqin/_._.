# 特征组装（新增：负载特征拼接、维度适配）
import torch
import numpy as np
from torch_geometric.data import Data, Batch


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

        # 创建 PyG Data 对象
        # 注意: req_vec 是全局特征，暂时挂在 data 对象上，或者拼接到 x 里(但这会破坏图结构)
        # 我们通常把它作为独立属性传递
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.req_vec = req_vec.unsqueeze(0)  # [1, D]

        return data

    def collate_fn(self, batch_data):
        """
        核心功能：将多个 (state, action, ...) 样本打包成 Batch
        供 DataLoader 或 ReplayBuffer 使用
        """
        batch_states = []
        batch_actions_high = []
        batch_actions_low = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_masks = []

        for item in batch_data:
            # item 结构取决于 ReplayBuffer 的存储格式
            # 假设: (state, action, reward, next_state, done, masks)
            state, action, reward, next_state, done, masks = item

            # 1. 处理当前状态
            batch_states.append(self.to_pyg_data(state))

            # 2. 处理下一状态
            batch_next_states.append(self.to_pyg_data(next_state))

            # 3. 其他标量数据
            high_act, low_act = action
            batch_actions_high.append(high_act)
            batch_actions_low.append(low_act)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_masks.append(masks)

        # === 核心：PyG Batch 拼接 ===
        # Batch.from_data_list 会自动处理 edge_index 的偏移
        batched_state = Batch.from_data_list(batch_states).to(self.device)
        batched_next_state = Batch.from_data_list(batch_next_states).to(self.device)

        # 处理全局特征 req_vec (需要手动拼接)
        # 从 [B, 1, D] -> [B, D]
        batched_state.req_vec = torch.cat([d.req_vec for d in batch_states], dim=0).to(self.device)
        batched_next_state.req_vec = torch.cat([d.req_vec for d in batch_next_states], dim=0).to(self.device)

        # 转换为 Tensor
        actions_high = torch.tensor(batch_actions_high, dtype=torch.long, device=self.device)
        actions_low = torch.tensor(batch_actions_low, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Mask 处理 (假设 Mask 是 numpy 数组)
        # mask[0] 是 high_mask, mask[1] 是 low_mask
        masks_high = torch.tensor(np.array([m[0] for m in batch_masks]), dtype=torch.float32, device=self.device)
        masks_low = torch.tensor(np.array([m[1] for m in batch_masks]), dtype=torch.float32, device=self.device)

        return {
            'state': batched_state,
            'next_state': batched_next_state,
            'action': (actions_high, actions_low),
            'reward': rewards,
            'done': dones,
            'mask': (masks_high, masks_low)
        }