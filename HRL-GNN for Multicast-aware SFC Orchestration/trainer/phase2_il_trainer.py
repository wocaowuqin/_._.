#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 Imitation Learning Trainer (修复路径匹配版)
✔ 保存到 outputs/checkpoints (系统默认读取位置)
✔ 文件名统一为 il_model_final.pth
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from pathlib import Path


class Phase2ILTrainer:
    def __init__(
            self,
            agent,
            expert_data_path: str,
            output_dir: str,
            config: dict
    ):
        self.agent = agent
        self.cfg = config

        # -------------------------------------------------
        # 🎯 关键修改：回归默认路径，确保 Phase 3 能找到
        # -------------------------------------------------
        # 强制保存到 checkpoints 目录
        base_output = Path(output_dir).parent
        if base_output.name == 'outputs':
            self.output_dir = base_output / "checkpoints"
        else:
            self.output_dir = Path(output_dir)  # 兜底

        # 创建目录
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = next(self.agent.policy_net.parameters()).device
        self.batch_size = config.get("batch_size", 32)
        self.request_dim = config.get("request_feat_dim", 24)
        self.epochs = config.get("epochs", 10)

        self.criterion = nn.CrossEntropyLoss()

        # 加载数据
        if not os.path.exists(expert_data_path):
            raise FileNotFoundError(f"Data file not found: {expert_data_path}")

        with open(expert_data_path, "rb") as f:
            raw_data = pickle.load(f)

        if isinstance(raw_data, dict):
            if 'success' in raw_data:
                self.expert_data = raw_data['success']
            else:
                self.expert_data = []
        elif isinstance(raw_data, list):
            self.expert_data = raw_data
        else:
            raise ValueError("Unknown data format")

        print(f"[Phase2] Loaded {len(self.expert_data)} expert samples")

        if len(self.expert_data) == 0:
            print("❌ Error: No samples loaded!")
            self.dataloader = []
        else:
            self.dataloader = DataLoader(
                self.expert_data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._collate_fn
            )

    def _collate_fn(self, batch):
        states = []
        actions = []
        req_vecs = []

        for item in batch:
            state = item.get("state", item.get("network_state"))
            if state is None: continue

            action = item.get("dest_idx", item.get("high_action", item.get("action")))

            states.append(state)
            actions.append(action)

            if hasattr(state, "req_vec"):
                req_vecs.append(state.req_vec)
            elif hasattr(state, "req"):
                req_vecs.append(state.req)
            else:
                req_vecs.append(torch.zeros(24))

        if not states:
            return None, None, None

        graph_batch = Batch.from_data_list(states)
        req_vecs = torch.stack(req_vecs, dim=0).float()
        actions = torch.tensor(actions, dtype=torch.long)

        return graph_batch, req_vecs, actions

    def run(self):
        if not self.dataloader:
            print("❌ No data loader ready.")
            return

        print("🚀 Starting Phase 2 Training...")
        print(f"📂 Model will be saved to: {self.output_dir} (matching system default)")

        self.agent.policy_net.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0

            for batch_data in self.dataloader:
                states, req_vecs, actions = batch_data
                if states is None: continue

                loss = self._compute_loss(states, req_vecs, actions)

                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(1, count)
            print(f"[Phase2][Epoch {epoch + 1}/{self.epochs}] Avg Loss = {avg_loss:.6f}")

        # 🎯 关键修改：保存为 il_model_final.pth
        self._save_model("il_model_final.pth")
        print("Phase 2 Complete.")

    def _compute_loss(self, states, req_vecs, target_actions):
        states = states.to(self.device)
        req_vecs = req_vecs.to(self.device)
        target_actions = target_actions.to(self.device)

        outputs = self.agent.policy_net(
            x=states.x,
            edge_index=states.edge_index,
            edge_attr=states.edge_attr,
            req_vec=req_vecs,
            batch=states.batch
        )

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        return self.criterion(logits, target_actions)

    def _save_model(self, name="il_model_final.pth"):
        path = self.output_dir / name
        torch.save(self.agent.policy_net.state_dict(), path)
        print(f"[Phase2] Model saved to {path}")