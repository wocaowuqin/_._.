#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 Imitation Learning Trainer (对齐配置文件版)

配置对齐：
1. ✅ 读取 phase2.yaml 的所有配置
2. ✅ 支持验证集划分
3. ✅ 实现早停策略
4. ✅ 多线程数据加载
5. ✅ 定期验证和保存
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ========================================
# 早停机制
# ========================================

class EarlyStopping:
    """早停策略"""

    def __init__(self, patience: int = 20, min_delta: float = 0.00001):
        """
        参数:
            patience: 容忍轮数
            min_delta: 最小改进阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停

        参数:
            val_loss: 验证损失

        返回:
            should_stop: bool
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # 如果改进不够
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"⚠️  EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"🛑 触发早停！验证损失 {self.patience} 轮未改进")
                return True
        else:
            # 有改进，重置计数器
            self.best_loss = val_loss
            self.counter = 0

        return False


# ========================================
# 数据集类
# ========================================

class ExpertDataset(Dataset):
    """Phase 2 专家数据集（支持数据转换）"""

    def __init__(self, expert_data_path: str):
        self.samples = []
        self._load_and_convert(expert_data_path)

    def _load_and_convert(self, data_path: str):
        """加载并转换数据"""

        logger.info("=" * 60)
        logger.info(f"📂 加载专家数据: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        # 提取 transitions
        if isinstance(raw_data, dict):
            transitions = raw_data.get('success', raw_data.get('data', []))
        elif isinstance(raw_data, list):
            transitions = raw_data
        else:
            raise ValueError(f"未知的数据格式: {type(raw_data)}")

        logger.info(f"✅ 原始数据: {len(transitions)} 个 Transition")

        if len(transitions) == 0:
            logger.error("❌ 没有可用的训练数据！")
            return

        # 转换数据
        converted = 0
        skipped = 0

        for i, trans in enumerate(transitions):
            try:
                action = trans.get('action')

                if isinstance(action, dict):
                    # Phase 1 格式：字典（需要转换）
                    converted_samples = self._convert_path_to_steps(trans)
                    self.samples.extend(converted_samples)
                    converted += len(converted_samples)

                elif isinstance(action, (int, np.integer)):
                    # 已经是单步格式
                    self.samples.append(trans)
                    converted += 1

                else:
                    skipped += 1

            except Exception as e:
                logger.warning(f"⚠️  样本 {i} 处理失败: {e}")
                skipped += 1

        logger.info(f"✅ 转换完成: {converted} 个样本, 跳过 {skipped} 个")
        logger.info(f"✅ 最终可用: {len(self.samples)} 个样本")
        logger.info("=" * 60)

    def _convert_path_to_steps(self, trans: Dict) -> List[Dict]:
        """将路径字典转换为单步动作序列"""
        action_dict = trans['action']
        path = action_dict.get('path', [])

        if not path or len(path) < 2:
            return []

        steps = []
        total_reward = trans.get('reward', 0)
        num_steps = len(path) - 1

        for step_idx in range(1, len(path)):
            node = int(path[step_idx]) if isinstance(path[step_idx], np.integer) else path[step_idx]

            # 🔥 关键修复：将 1-based 转换为 0-based
            # Expert 数据是 [1, 28]，但模型期望 [0, 27]
            if node >= 1 and node <= 28:
                node = node - 1  # 转换为 0-based

            # 分配奖励
            step_reward = total_reward * 0.5 if step_idx == len(path) - 1 else total_reward * 0.5 / max(1,
                                                                                                        num_steps - 1)

            step_trans = {
                'state': trans.get('state'),
                'network_state': trans.get('network_state', trans.get('state')),
                'action': node,
                'dest_idx': node,
                'high_action': node,
                'reward': step_reward,
                'next_state': trans.get('next_state'),
                'done': (step_idx == len(path) - 1),
                'masks': trans.get('masks')
            }

            steps.append(step_trans)

        return steps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ========================================
# Phase 2 Trainer (完整对齐版)
# ========================================

class Phase2ILTrainer:
    """Phase 2 模仿学习训练器（完全对齐 phase2.yaml）"""

    def __init__(
            self,
            agent,
            expert_data_path: str,
            output_dir: str,
            config: dict
    ):
        self.agent = agent
        self.cfg = config

        # ========================================
        # 1. 读取 phase2.yaml 配置
        # ========================================
        phase2_cfg = config.get('phase2', {})

        # 基础配置
        self.epochs = phase2_cfg.get('epochs', 100)
        self.batch_size = phase2_cfg.get('batch_size', 128)
        self.save_every_epoch = phase2_cfg.get('save_every_epoch', 2)
        self.validation_split = phase2_cfg.get('validation_split', 0.1)

        # 数据加载配置
        data_loader_cfg = phase2_cfg.get('data_loader', {})
        self.shuffle = data_loader_cfg.get('shuffle', True)
        self.num_workers = data_loader_cfg.get('num_workers', 4)
        self.pin_memory = data_loader_cfg.get('pin_memory', True)

        # 🔥 Windows 兼容性修复
        import platform
        if platform.system() == 'Windows':
            if self.num_workers > 0:
                logger.warning("⚠️  Windows 系统检测到，强制禁用多进程加载")
                logger.warning("   num_workers: 4 → 0")
                self.num_workers = 0
            if self.pin_memory:
                logger.warning("   pin_memory: True → False")
                self.pin_memory = False

        # 早停配置
        early_stopping_cfg = phase2_cfg.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_cfg.get('patience', 20),
            min_delta=early_stopping_cfg.get('min_delta', 0.0001)
        )

        # ========================================
        # 2. 设置输出路径
        # ========================================
        # 优先使用配置中的路径
        if 'output_dir' in phase2_cfg:
            self.output_dir = Path(phase2_cfg['output_dir'])
        else:
            base_output = Path(output_dir).parent
            if base_output.name == 'outputs':
                self.output_dir = base_output / "checkpoints"
            else:
                self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ========================================
        # 3. 日志输出配置
        # ========================================
        logger.info("=" * 70)
        logger.info("Phase 2 Imitation Learning Trainer (配置对齐版)")
        logger.info("=" * 70)
        logger.info(f"📋 训练配置:")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Batch Size: {self.batch_size}")
        logger.info(f"  - Validation Split: {self.validation_split * 100}%")
        logger.info(f"  - Save Every: {self.save_every_epoch} epochs")
        logger.info(f"  - Shuffle: {self.shuffle}")
        logger.info(f"  - Num Workers: {self.num_workers}")
        logger.info(f"  - Pin Memory: {self.pin_memory}")
        logger.info(f"📋 早停配置:")
        logger.info(f"  - Patience: {early_stopping_cfg.get('patience', 3)}")
        logger.info(f"  - Min Delta: {early_stopping_cfg.get('min_delta', 0.001)}")
        logger.info(f"📂 输出目录: {self.output_dir}")
        logger.info("=" * 70)

        # ========================================
        # 4. 获取模型和优化器
        # ========================================
        if hasattr(agent, 'policy_net'):
            self.model = agent.policy_net
            self.optimizer = agent.optimizer
            logger.info("✅ 使用 Phase 2 Agent (policy_net)")
        elif hasattr(agent, 'q_high_online'):
            self.model = agent.q_high_online
            self.optimizer = getattr(agent, 'optimizer_high', agent.optimizer)
            logger.warning("⚠️  检测到 Phase 3 Agent，使用 High-Level 网络")
        elif hasattr(agent, 'q_network'):
            self.model = agent.q_network
            self.optimizer = agent.optimizer
            logger.info("✅ 使用通用 DQN Agent (q_network)")
        else:
            raise RuntimeError("❌ Agent 缺少可识别的网络属性")

        self.device = next(self.model.parameters()).device
        logger.info(f"🖥️  训练设备: {self.device}")

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # ========================================
        # 5. 加载数据集并划分训练/验证集
        # ========================================
        logger.info(f"📂 专家数据路径: {expert_data_path}")

        full_dataset = ExpertDataset(expert_data_path)

        if len(full_dataset) == 0:
            logger.error("❌ 数据集为空！")
            self.train_loader = None
            self.val_loader = None
            return

        # 划分训练集和验证集
        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"📊 数据集划分:")
        logger.info(f"  - 训练集: {train_size} 个样本")
        logger.info(f"  - 验证集: {val_size} 个样本")

        # 创建 DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False
        )

        logger.info(f"✅ DataLoader 创建成功:")
        logger.info(f"  - 训练 Batches: {len(self.train_loader)}")
        logger.info(f"  - 验证 Batches: {len(self.val_loader)}")
        logger.info("=" * 70)

        # 请求特征维度（从配置或默认值）
        self.request_dim = config.get('request_feat_dim', 24)

    def _collate_fn(self, batch: List[Dict]) -> Optional[Tuple]:
        """数据批处理函数"""
        states = []
        actions = []
        req_vecs = []

        for item in batch:
            state = item.get('state') or item.get('network_state')
            if state is None:
                continue

            action = item.get('action') or item.get('dest_idx') or item.get('high_action')
            if action is None:
                continue

            if isinstance(action, np.integer):
                action = int(action)

            states.append(state)
            actions.append(action)

            if hasattr(state, 'req_vec'):
                req_vecs.append(state.req_vec)
            elif hasattr(state, 'req'):
                req_vecs.append(state.req)
            else:
                req_vecs.append(torch.zeros(self.request_dim))

        if not states:
            return None

        try:
            graph_batch = Batch.from_data_list(states)
            req_vecs = torch.stack(req_vecs, dim=0).float()
            actions = torch.tensor(actions, dtype=torch.long)

            return graph_batch, req_vecs, actions
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            return None

    def run(self):
        """运行训练（对齐 phase2.yaml 配置）"""

        if self.train_loader is None:
            logger.error("❌ DataLoader 未就绪，无法训练")
            return

        logger.info("🚀 开始 Phase 2 训练...")

        self.model.train()

        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            # ========================================
            # 训练阶段
            # ========================================
            train_loss = self._train_epoch(epoch)

            # ========================================
            # 验证阶段
            # ========================================
            val_loss = self._validate_epoch(epoch)

            # ========================================
            # 日志输出
            # ========================================
            logger.info(
                f"Epoch {epoch:2d}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # ========================================
            # 保存最佳模型
            # ========================================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                logger.info(f"🏆 新的最佳验证损失: {val_loss:.6f}")

            # ========================================
            # 定期保存
            # ========================================
            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)

            # ========================================
            # 早停检查
            # ========================================
            if self.early_stopping(val_loss):
                logger.info(f"🛑 早停触发，训练终止于 Epoch {epoch}")
                break

        # ========================================
        # 最终保存
        # ========================================
        self.save_checkpoint(epoch, train_loss, val_loss, is_final=True)

        logger.info("=" * 70)
        logger.info("✅ Phase 2 训练完成")
        logger.info(f"  最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"  模型保存位置: {self.output_dir / 'il_model_final.pth'}")
        logger.info("=" * 70)

    def _train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        batch_count = 0

        # 使用 tqdm 显示进度
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            leave=False
        )

        for batch_data in pbar:
            if batch_data is None:
                continue

            states, req_vecs, actions = batch_data

            # 计算损失
            loss = self._compute_loss(states, req_vecs, actions)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(1, batch_count)

    def _validate_epoch(self, epoch: int) -> float:
        """验证一个 epoch"""
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                if batch_data is None:
                    continue

                states, req_vecs, actions = batch_data

                # 计算损失
                loss = self._compute_loss(states, req_vecs, actions)

                total_loss += loss.item()
                batch_count += 1

        return total_loss / max(1, batch_count)

    def _compute_loss(
            self,
            states: Batch,
            req_vecs: torch.Tensor,
            target_actions: torch.Tensor
    ) -> torch.Tensor:
        """计算损失"""
        states = states.to(self.device)
        req_vecs = req_vecs.to(self.device)
        target_actions = target_actions.to(self.device)

        # 前向传播
        outputs = self.model(
            x=states.x,
            edge_index=states.edge_index,
            edge_attr=states.edge_attr if hasattr(states, 'edge_attr') else None,
            req_vec=req_vecs,
            batch=states.batch
        )

        # 提取 logits
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # 计算损失
        loss = self.criterion(logits, target_actions)

        return loss

    def save_checkpoint(
            self,
            epoch: int,
            train_loss: float,
            val_loss: float,
            is_best: bool = False,
            is_final: bool = False
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'policy_net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.cfg
        }

        # 定期保存
        if not is_best and not is_final:
            ckpt_path = self.output_dir / f"il_model_epoch_{epoch}.pth"
            torch.save(checkpoint, ckpt_path)
            logger.info(f"💾 Checkpoint 已保存: epoch_{epoch}.pth")

        # 最佳模型
        if is_best:
            best_path = self.output_dir / "il_model_best.pth"
            torch.save(checkpoint, best_path)

        # 最终模型
        if is_final:
            final_path = self.output_dir / "il_model_final.pth"
            torch.save(checkpoint, final_path)
            logger.info(f"✅ 最终模型已保存: il_model_final.pth")
