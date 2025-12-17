# core/hrl/agent.py
"""
统一的 HRL Agent 实现
整合了 Agent_wrapper.py, agent_gnn.py 和 HierarchicalPolicy
"""

import torch
import torch.optim as optim
import numpy as np
import random
import logging
from collections import deque

# 导入层次化策略网络
from core.hrl.high_policy import HierarchicalPolicy

# 导入特征构建器（用于 Batch 处理）
from core.gnn.feature_builder import GNNFeatureBuilder

logger = logging.getLogger(__name__)


class HRL_DQN_Agent:
    """
    分层强化学习智能体

    整合特性:
    1. GNNFeatureBuilder - 处理图数据的批处理
    2. HierarchicalPolicy - 包含 GNN + MidPolicy + LowPolicy
    3. Double DQN 训练机制
    4. Epsilon-Greedy 探索
    """

    def __init__(self, config):
        """
        从配置初始化 Agent

        Args:
            config: 配置字典，需要包含:
                - gnn: GNN配置
                - env: 环境配置
                - training: 训练配置
                - eval: 评估配置
                - epsilon: 探索配置
        """
        self.cfg = config
        self.device = torch.device(config['eval']['device'])

        # === 1. 初始化策略网络 ===
        # HierarchicalPolicy 内部包含: GNN + MidPolicy + LowPolicy
        self.policy_net = HierarchicalPolicy(config).to(self.device)
        self.target_net = HierarchicalPolicy(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # === 2. 初始化优化器 ===
        lr = float(config['training']['learning_rate'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # === 3. 初始化特征构建器 ===
        # 负责把环境的 tuple 数据转为 PyG Batch
        self.feature_builder = GNNFeatureBuilder(self.device)

        # === 4. 经验回放池 ===
        buffer_size = int(config['training']['buffer_size'])
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = int(config['training']['batch_size'])
        self.gamma = float(config['training']['gamma'])

        # === 5. Epsilon-Greedy 参数 ===
        epsilon_cfg = config.get('epsilon', {})
        self.epsilon = float(epsilon_cfg.get('initial', 1.0))
        self.epsilon_end = float(epsilon_cfg.get('final', 0.01))
        self.epsilon_decay = float(epsilon_cfg.get('decay_steps', 10000))
        self.steps_done = 0

        # === 6. 动作空间大小（用于兼容性）===
        env_cfg = config.get('env', {})
        self.n_actions = env_cfg.get('nb_low_level_actions', 50)
        self.n_goals = env_cfg.get('nb_high_level_goals', 10)

        # === 7. 训练状态 ===
        self._training = True
        self.update_count = 0
        self.target_update_freq = int(config['training'].get('target_update_freq', 1000))

        logger.info(f"✅ HRL_DQN_Agent 初始化完成")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   高层动作: {self.n_goals}")
        logger.info(f"   低层动作: {self.n_actions}")
        logger.info(f"   学习率: {lr}")
        logger.info(f"   Batch大小: {self.batch_size}")

    def get_epsilon(self):
        """计算当前 epsilon 值"""
        return self.epsilon_end + (self.epsilon_end - self.epsilon) * \
            np.exp(-1. * self.steps_done / max(1, self.epsilon_decay))

    def select_action(self, state, masks, epsilon=None, expert_action=None, beta=0.0):
        """
        动作选择（支持 Epsilon-Greedy 和 DAgger）

        Args:
            state: (x, edge_index, edge_attr, req_vec) 来自环境
            masks: (high_mask, low_mask) 来自 PolicyHelper
            epsilon: 可选的 epsilon 值（如果为 None，使用衰减值）
            expert_action: 可选的专家动作 (high_act, low_act)
            beta: DAgger 混合系数

        Returns:
            (high_action, low_action) 元组
        """
        high_mask, low_mask = masks
        self.steps_done += 1

        # === 1. DAgger 专家引导 ===
        if expert_action is not None and random.random() < beta:
            high_act, low_act = expert_action
            # 检查动作合法性
            if high_mask[high_act] > 0 and low_mask[low_act] > 0:
                return int(high_act), int(low_act)

        # === 2. Epsilon-Greedy 探索 ===
        if epsilon is None:
            epsilon = self.get_epsilon()

        if random.random() < epsilon:
            # 随机选择合法动作
            high_valid = np.where(high_mask > 0)[0]
            low_valid = np.where(low_mask > 0)[0]

            if len(high_valid) == 0 or len(low_valid) == 0:
                logger.warning("⚠️  没有合法动作，返回默认值 (0, 0)")
                return 0, 0

            high_act = int(np.random.choice(high_valid))
            low_act = int(np.random.choice(low_valid))
            return high_act, low_act

        # === 3. 策略网络选择（Exploitation）===
        with torch.no_grad():
            # 转换为 PyG 数据
            pyg_data = self.feature_builder.to_pyg_data(state)

            # 移动到设备
            x = pyg_data.x.to(self.device)
            edge_index = pyg_data.edge_index.to(self.device)
            edge_attr = pyg_data.edge_attr.to(self.device)
            req_vec = pyg_data.req_vec.to(self.device)

            # 前向传播
            mid_logits, low_logits, _, _ = self.policy_net(
                x, edge_index, edge_attr, req_vec
            )

            # 应用动作掩码
            t_high_mask = torch.tensor(high_mask, dtype=torch.float32, device=self.device)
            t_low_mask = torch.tensor(low_mask, dtype=torch.float32, device=self.device)

            mid_logits[0, t_high_mask == 0] = -1e9
            low_logits[0, t_low_mask == 0] = -1e9

            # Argmax 选择
            high_act = int(mid_logits.argmax(dim=1).item())
            low_act = int(low_logits.argmax(dim=1).item())

        return high_act, low_act

    def store_transition(self, state, action, reward, next_state, done,
                         goal=None, next_valid_mask=None):
        """
        存储经验到回放池

        Args:
            state: 当前状态
            action: (high_action, low_action) 元组
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
            goal: 高层目标（可选）
            next_valid_mask: 下一状态的动作掩码（可选）
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            'next_valid_mask': next_valid_mask
        }
        self.memory.append(transition)

    def update(self):
        """
        训练步骤：采样 -> 批处理 -> 计算损失 -> 反向传播

        Returns:
            loss: 损失值（float）
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # === 1. 采样 ===
        transitions = random.sample(self.memory, self.batch_size)

        # === 2. 使用 Feature Builder 组装 Batch ===
        try:
            batch = self.feature_builder.collate_fn(transitions)
        except Exception as e:
            logger.error(f"❌ Batch 组装失败: {e}")
            return 0.0

        # === 3. 提取数据 ===
        state_batch = batch['state']
        next_state_batch = batch['next_state']
        actions = batch['action']  # (high_actions, low_actions)
        rewards = batch['reward']
        dones = batch['done']

        # 分离高层和低层动作
        if isinstance(actions, tuple):
            actions_high, actions_low = actions
        else:
            # 假设 actions 是 tensor [B, 2]
            actions_high = actions[:, 0]
            actions_low = actions[:, 1]

        # === 4. 计算当前 Q 值（Online Network）===
        curr_mid_q, curr_low_q, _, _ = self.policy_net(
            state_batch.x,
            state_batch.edge_index,
            state_batch.edge_attr,
            state_batch.req_vec,
            batch=state_batch.batch
        )

        # Gather 选定动作的 Q 值
        q_high = curr_mid_q.gather(1, actions_high.unsqueeze(1))
        q_low = curr_low_q.gather(1, actions_low.unsqueeze(1))

        # === 5. 计算目标 Q 值（Target Network）===
        with torch.no_grad():
            next_mid_q, next_low_q, _, _ = self.target_net(
                next_state_batch.x,
                next_state_batch.edge_index,
                next_state_batch.edge_attr,
                next_state_batch.req_vec,
                batch=next_state_batch.batch
            )

            # Max Q 值
            max_next_high = next_mid_q.max(1)[0].unsqueeze(1)
            max_next_low = next_low_q.max(1)[0].unsqueeze(1)

            # Bellman 目标
            target_high = rewards + (1 - dones) * self.gamma * max_next_high
            target_low = rewards + (1 - dones) * self.gamma * max_next_low

        # === 6. 计算损失 ===
        loss_high = torch.nn.functional.mse_loss(q_high, target_high)
        loss_low = torch.nn.functional.mse_loss(q_low, target_low)
        total_loss = loss_high + loss_low

        # === 7. 反向传播 ===
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # === 8. 更新 Target Network ===
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"🔄 Target Network 更新 (step {self.update_count})")

        # === 9. Epsilon 衰减 ===
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
            )

        return total_loss.item()

    def train(self):
        """设置为训练模式"""
        self._training = True
        self.policy_net.train()

    def eval(self):
        """设置为评估模式"""
        self._training = False
        self.policy_net.eval()

    def save(self, path):
        """
        保存模型

        Args:
            path: 保存路径
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'update_count': self.update_count
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 模型已保存到: {path}")

    def load(self, path):
        """
        加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']

        logger.info(f"📥 模型已加载: {path}")
        logger.info(f"   Epsilon: {self.epsilon:.4f}")
        logger.info(f"   Steps: {self.steps_done}")


def create_agent_from_config(config):
    """
    工厂函数：从配置创建 Agent
    （提供向后兼容性）

    Args:
        config: 配置字典

    Returns:
        HRL_DQN_Agent 实例
    """
    return HRL_DQN_Agent(config)


# 向后兼容的别名
Agent = HRL_DQN_Agent

if __name__ == "__main__":
    # 测试用例
    test_config = {
        'gnn': {
            'node_feat_dim': 10,
            'edge_feat_dim': 3,
            'request_feat_dim': 6,
            'hidden_dim': 128,
            'num_gat_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        },
        'env': {
            'nb_high_level_goals': 10,
            'nb_low_level_actions': 50
        },
        'training': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 32,
            'target_update_freq': 1000
        },
        'eval': {
            'device': 'cpu'
        },
        'epsilon': {
            'initial': 1.0,
            'final': 0.01,
            'decay_steps': 10000
        }
    }

    try:
        agent = HRL_DQN_Agent(test_config)
        print(f"✅ Agent 创建成功")
        print(f"   动作空间: {agent.n_actions}")
        print(f"   目标空间: {agent.n_goals}")
        print(f"   设备: {agent.device}")
    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        import traceback

        traceback.print_exc()