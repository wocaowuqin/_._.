import torch
import torch.nn as nn
import logging
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 根据你的数据格式解包
        item = self.data[idx]
        return item['state'], item['high_action']


class Phase2ILTrainer:
    """
    阶段 2：模仿学习训练器
    """

    def __init__(self, agent, expert_data_path, output_dir, config):
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        # 加载数据
        if isinstance(expert_data_path, list):
            self.raw_data = expert_data_path
        else:
            with open(expert_data_path, 'rb') as f:
                self.raw_data = pickle.load(f)

        self.dataset = ExpertDataset(self.raw_data)
        self.dataloader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.agent.policy_net.parameters(), lr=1e-4)

    def run(self):
        logger.info("🚀 Starting Phase 2: Imitation Learning...")
        epochs = self.cfg['epochs']

        for epoch in range(epochs):
            total_loss = 0
            self.agent.policy_net.train()

            for batch_idx, (states, actions) in enumerate(self.dataloader):
                # 1. 数据搬运 (假设是 GNN 格式，这里需要特殊的 collate_fn，此处简化为 Flat)
                # 如果是 GNN，state 是个 tuple (x, edge_index...)，需要拆解
                # 此处代码需根据 Agent 的输入接口微调

                # 假设 Agent 有一个 supervised_loss 方法
                loss = self._compute_loss(states, actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.cfg.get('save_every', 2) == 0:
                self.agent.save(self.output_dir / f"il_model_ep{epoch + 1}.pth")

        self.agent.save(self.output_dir / "il_model_final.pth")
        logger.info("Phase 2 Complete.")

    def _compute_loss(self, states, target_actions):
        """
        计算模仿学习损失
        这部分逻辑通常放在 Agent 内部更好，这里为了独立演示写在这里
        """
        # 这里的实现高度依赖于 state 的数据结构
        # 假设 agent.policy_net 返回 logits
        # logits, _, _ = self.agent.policy_net(states...)
        # return self.criterion(logits, target_actions)
        return torch.tensor(0.0, requires_grad=True)  # 占位