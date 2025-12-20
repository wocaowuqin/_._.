"""
core/hrl/agent.py - Unified HRL Agent
修复记录:
1. ✅ load() 方法增加对 Phase 2 模型 (policy_net) 的兼容支持
2. ✅ __init__ 接收 kwargs，防止 TypeError
3. ✅ 优先使用环境传入的维度，防止 Invalid Action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging
from collections import deque
import os

# Import hierarchical policy network
from core.hrl.high_policy import HierarchicalPolicy

# Import feature builder
from core.gnn.feature_builder import GNNFeatureBuilder

logger = logging.getLogger(__name__)


class QNetworkWrapper(nn.Module):
    """
    Phase 3 专用 Q-network (内嵌冻结的 Encoder)
    """

    def __init__(self, encoder, high_dim: int, low_dim: int, device, hidden_dim: int = 128):
        super().__init__()
        self.device = device

        # 1. 冻结 Encoder
        self.encoder = encoder
        self._freeze_encoder()

        # 2. 自动探测维度
        encoder_output_dim = self._probe_encoder_dim()

        # 3. High-level Q-head
        self.q_high = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, high_dim)
        )

        # 4. Low-level Q-head
        self.q_low = nn.Sequential(
            nn.Linear(encoder_output_dim + high_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, low_dim)
        )

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _probe_encoder_dim(self):
        try:
            # 构造最小 Dummy Batch 进行探测
            dummy_x = torch.randn(2, 17).to(self.device)
            dummy_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
            dummy_edge_attr = torch.randn(2, 5).to(self.device)
            dummy_req = torch.randn(1, 24).to(self.device)
            dummy_batch = torch.zeros(2, dtype=torch.long).to(self.device)

            with torch.no_grad():
                out = self.encoder(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_req, dummy_batch)
            return out.shape[-1]
        except Exception as e:
            logger.warning(f"[Probe] 探测失败，使用默认值 128: {e}")
            return 128

    def get_high_q_values(self, x, edge_index, edge_attr, req_vec, batch=None):
        z = self.encoder(x, edge_index, edge_attr, req_vec, batch=batch)
        return self.q_high(z)

    def get_low_q_values(self, x, edge_index, edge_attr, req_vec, high_action_onehot, batch=None):
        z = self.encoder(x, edge_index, edge_attr, req_vec, batch=batch)
        low_state = torch.cat([z, high_action_onehot], dim=1)
        return self.q_low(low_state)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


class HRL_DQN_Agent:
    """
    分层强化学习智能体
    """

    def __init__(self, config, phase=3, **kwargs):
        """
        初始化 Agent
        ✅ kwargs 接收 'high_action_dim', 'low_action_dim' 等参数
        """
        self.cfg = config
        self.phase = int(phase)
        self.device = torch.device(config['eval']['device'])

        # 🔥【关键修复】确定动作空间
        env_cfg = config.get('environment', config.get('env', {}))

        # Low Action Dim
        if 'low_action_dim' in kwargs:
            self.n_actions = int(kwargs['low_action_dim'])
        else:
            self.n_actions = env_cfg.get('nb_low_level_actions', 50)

        # High Action Dim
        if 'high_action_dim' in kwargs:
            self.n_goals = int(kwargs['high_action_dim'])
        else:
            self.n_goals = env_cfg.get('nb_high_level_goals', 10)

        logger.info(f"🔧 [Agent] Action Space: High={self.n_goals}, Low={self.n_actions}")
        logger.info(f"Initializing HRL Agent for Phase {self.phase}...")

        if self.phase == 2:
            self._init_phase2()
        else:
            self._init_phase3()

        # 经验回放
        buffer_size = int(config['training']['buffer_size'])
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = int(config['training']['batch_size'])
        self.gamma = float(config['training']['gamma'])

        # Epsilon 参数
        epsilon_cfg = config.get('epsilon', {})
        self.epsilon_start = float(epsilon_cfg.get('initial', 1.0))
        self.epsilon = self.epsilon_start
        self.epsilon_end = float(epsilon_cfg.get('final', 0.01))
        self.epsilon_decay = float(epsilon_cfg.get('decay_steps', 10000))
        self.steps_done = 0

        self.feature_builder = GNNFeatureBuilder(self.device)
        self._training = True
        self.update_count = 0
        self.target_update_freq = int(config['training'].get('target_update_freq', 1000))

        logger.info(f"✅ HRL_DQN_Agent Initialized (Phase {self.phase})")

    def __getattr__(self, name):
        if name == 'policy_net' and self.phase == 3:
            raise RuntimeError("❌ [Fatal Error] Phase 3 禁止调用 'policy_net'! 请检查代码逻辑。")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _init_phase2(self):
        """Phase 2 Init"""
        self.policy_net = HierarchicalPolicy(self.cfg).to(self.device)

        encoder_path = self.cfg.get('gnn', {}).get('encoder_path', "outputs/il_train/shared_encoder.pth")
        if encoder_path and os.path.exists(encoder_path):
            try:
                target_encoder = None
                if hasattr(self.policy_net, 'gnn') and hasattr(self.policy_net.gnn, 'encoder'):
                    target_encoder = self.policy_net.gnn.encoder
                if target_encoder:
                    target_encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
                    logger.info(f"[Phase2] Loaded Shared Encoder: {encoder_path}")
            except Exception as e:
                logger.error(f"[Phase2] Failed to load encoder: {e}")

        self.target_net = HierarchicalPolicy(self.cfg).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        lr = float(self.cfg['training']['learning_rate'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def _init_phase3(self):
        """Phase 3 Init"""
        logger.info("[Phase3] 🚀 Initializing RL Architecture")
        temp_model = HierarchicalPolicy(self.cfg).to(self.device)
        shared_enc_path = "outputs/il_train/shared_encoder.pth"
        full_model_path = "outputs/checkpoints/il_model_final.pth"

        encoder = None
        # 1. 优先尝试从 IL 完整模型提取 Encoder
        if os.path.exists(full_model_path):
            try:
                checkpoint = torch.load(full_model_path, map_location=self.device)
                key = 'policy_net' if 'policy_net' in checkpoint else None
                state_dict = checkpoint[key] if key else checkpoint
                temp_model.load_state_dict(state_dict, strict=False)
                encoder = temp_model.gnn.encoder
                logger.info(f"[Phase3] Extracted Encoder from {full_model_path}")
            except Exception as e:
                logger.warning(f"[Phase3] Failed to extract from full model: {e}")

        # 2. 其次尝试单独的 encoder
        if encoder is None and os.path.exists(shared_enc_path):
            try:
                state_dict = torch.load(shared_enc_path, map_location=self.device)
                temp_model.gnn.encoder.load_state_dict(state_dict)
                encoder = temp_model.gnn.encoder
                logger.info(f"[Phase3] Loaded Shared Encoder from {shared_enc_path}")
            except Exception:
                pass

        if encoder is None:
            logger.warning("[Phase3] ⚠️ No pretrained encoder found! Training from scratch.")
            encoder = temp_model.gnn.encoder

        self.q_network = QNetworkWrapper(
            encoder=encoder,
            high_dim=self.n_goals,
            low_dim=self.n_actions,
            device=self.device,
            hidden_dim=128
        ).to(self.device)

        target_temp_model = HierarchicalPolicy(self.cfg).to(self.device)
        target_encoder = target_temp_model.gnn.encoder
        target_encoder.load_state_dict(encoder.state_dict())

        self.target_q_network = QNetworkWrapper(
            encoder=target_encoder,
            high_dim=self.n_goals,
            low_dim=self.n_actions,
            device=self.device,
            hidden_dim=128
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        lr = float(self.cfg['training']['learning_rate'])
        trainable_params = list(self.q_network.q_high.parameters()) + list(self.q_network.q_low.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=lr)

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / max(1, self.epsilon_decay))

    def select_action(self, state, masks, expert_action=None, beta=0.0, epsilon=None):
        """Action Selection"""
        high_mask, low_mask = masks
        self.steps_done += 1

        # 1. DAgger
        if expert_action is not None and random.random() < beta:
            high_act, low_act = expert_action
            if high_mask[high_act] > 0 and low_mask[low_act] > 0:
                return int(high_act), int(low_act)

        # 2. Epsilon-Greedy
        if epsilon is None:
            self.epsilon = self.get_epsilon()
            epsilon = self.epsilon

        if random.random() < epsilon:
            high_valid = np.where(high_mask > 0)[0]
            low_valid = np.where(low_mask > 0)[0]
            return (int(np.random.choice(high_valid)) if len(high_valid) > 0 else 0,
                    int(np.random.choice(low_valid)) if len(low_valid) > 0 else 0)

        # 3. Model Exploitation
        pyg_data = self.feature_builder.to_pyg_data(state)
        x = pyg_data.x.to(self.device)
        edge_index = pyg_data.edge_index.to(self.device)
        edge_attr = pyg_data.edge_attr.to(self.device)
        req_vec = pyg_data.req_vec.to(self.device)
        batch = getattr(pyg_data, "batch", None)

        with torch.no_grad():
            if self.phase == 2:
                outputs = self.policy_net(x, edge_index, edge_attr, req_vec, batch=batch)
                if isinstance(outputs, tuple):
                    mid_logits, low_logits = outputs[:2]
                else:
                    mid_logits, low_logits = outputs, outputs

                t_high_mask = torch.tensor(high_mask, dtype=torch.float32, device=self.device)
                t_low_mask = torch.tensor(low_mask, dtype=torch.float32, device=self.device)

                mid_logits[0, t_high_mask == 0] = -1e9
                low_logits[0, t_low_mask == 0] = -1e9

                high_act = int(mid_logits.argmax(dim=1).item())
                low_act = int(low_logits.argmax(dim=1).item())

            else:
                q_high = self.q_network.get_high_q_values(x, edge_index, edge_attr, req_vec, batch=batch)

                t_high_mask = torch.tensor(high_mask, dtype=torch.float32, device=self.device)
                q_high[0, t_high_mask == 0] = -1e9
                high_act = int(q_high.argmax(dim=1).item())

                high_action_onehot = torch.zeros(1, self.n_goals, device=self.device)
                high_action_onehot[0, high_act] = 1.0

                q_low = self.q_network.get_low_q_values(
                    x, edge_index, edge_attr, req_vec, high_action_onehot, batch=batch
                )

                t_low_mask = torch.tensor(low_mask, dtype=torch.float32, device=self.device)
                q_low[0, t_low_mask == 0] = -1e9
                low_act = int(q_low.argmax(dim=1).item())

        return high_act, low_act

    def store_transition(self, state, action, reward, next_state, done,
                         goal=None, next_valid_mask=None):
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

    def store(self, state, action, reward, next_state, done, goal, next_valid_actions=None):
        self.store_transition(state, action, reward, next_state, done, goal, next_valid_actions)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        transitions = random.sample(self.memory, self.batch_size)

        try:
            batch = self.feature_builder.collate_fn(transitions)
        except Exception as e:
            logger.error(f"❌ Batch assembly failed: {e}")
            return 0.0

        if self.phase == 2:
            return self._phase2_update(batch)
        else:
            return self._phase3_update(batch)

    def _phase2_update(self, batch):
        state_batch = batch['state']
        actions = batch['action']

        if isinstance(actions, tuple):
            actions_high, actions_low = actions
        elif isinstance(actions, torch.Tensor):
            actions_high = actions[:, 0]
            actions_low = actions[:, 1]
        else:
            actions_high = torch.tensor([a[0] for a in actions], device=self.device)
            actions_low = torch.tensor([a[1] for a in actions], device=self.device)

        outputs = self.policy_net(
            state_batch.x, state_batch.edge_index, state_batch.edge_attr,
            state_batch.req_vec, batch=state_batch.batch
        )
        if isinstance(outputs, tuple):
            curr_mid_q, curr_low_q = outputs[:2]
        else:
            curr_mid_q, curr_low_q = outputs, outputs

        loss_high = F.cross_entropy(curr_mid_q, actions_high)
        loss_low = F.cross_entropy(curr_low_q, actions_low)
        total_loss = loss_high + loss_low

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()

    def _phase3_update(self, batch):
        state_batch = batch['state']
        next_state_batch = batch['next_state']
        actions = batch['action']
        rewards = batch['reward']
        dones = batch['done']

        if isinstance(actions, tuple):
            high_actions, low_actions = actions
        elif isinstance(actions, torch.Tensor):
            high_actions = actions[:, 0]
            low_actions = actions[:, 1]
        else:
            high_actions = torch.tensor([a[0] for a in actions], device=self.device)
            low_actions = torch.tensor([a[1] for a in actions], device=self.device)

        # High Q
        q_high = self.q_network.get_high_q_values(
            state_batch.x, state_batch.edge_index, state_batch.edge_attr,
            state_batch.req_vec, state_batch.batch
        )
        q_high_selected = q_high.gather(1, high_actions.unsqueeze(1))

        with torch.no_grad():
            next_q_high_online = self.q_network.get_high_q_values(
                next_state_batch.x, next_state_batch.edge_index, next_state_batch.edge_attr,
                next_state_batch.req_vec, next_state_batch.batch
            )
            next_high_actions = next_q_high_online.argmax(dim=1, keepdim=True)

            next_q_high_target = self.target_q_network.get_high_q_values(
                next_state_batch.x, next_state_batch.edge_index, next_state_batch.edge_attr,
                next_state_batch.req_vec, next_state_batch.batch
            )
            q_high_target = next_q_high_target.gather(1, next_high_actions)
            q_high_target = rewards + self.gamma * q_high_target * (1 - dones)

        high_loss = F.smooth_l1_loss(q_high_selected, q_high_target)

        # Low Q
        high_actions_onehot = F.one_hot(high_actions, num_classes=self.n_goals).float()
        q_low = self.q_network.get_low_q_values(
            state_batch.x, state_batch.edge_index, state_batch.edge_attr,
            state_batch.req_vec, high_actions_onehot, state_batch.batch
        )
        q_low_selected = q_low.gather(1, low_actions.unsqueeze(1))

        with torch.no_grad():
            next_high_actions_onehot = F.one_hot(next_high_actions.squeeze(1),
                                                 num_classes=self.n_goals).float()
            next_q_low_online = self.q_network.get_low_q_values(
                next_state_batch.x, next_state_batch.edge_index, next_state_batch.edge_attr,
                next_state_batch.req_vec, next_high_actions_onehot, next_state_batch.batch
            )
            next_low_actions = next_q_low_online.argmax(dim=1, keepdim=True)
            q_low_target = next_q_low_online.gather(1, next_low_actions)
            q_low_target = rewards + self.gamma * q_low_target * (1 - dones)

        low_loss = F.smooth_l1_loss(q_low_selected, q_low_target)
        total_loss = high_loss + low_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return total_loss.item()

    def train(self):
        self._training = True
        if self.phase == 2:
            self.policy_net.train()
        else:
            self.q_network.train()

    def eval(self):
        self._training = False
        if self.phase == 2:
            self.policy_net.eval()
        else:
            self.q_network.eval()

    def save(self, path):
        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'phase': self.phase,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }
        if self.phase == 2:
            checkpoint['policy_net'] = self.policy_net.state_dict()
        else:
            checkpoint['q_network'] = self.q_network.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"💾 Model saved to: {path}")

    def load(self, path):
        """
        🔥 修复：智能加载方法 (支持从 IL 迁移到 RL)
        """
        logger.info(f"🔍 Loading model from {path} (Phase {self.phase})")
        checkpoint = torch.load(path, map_location=self.device)

        self.phase = checkpoint.get('phase', self.phase)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps_done = checkpoint.get('steps_done', self.steps_done)

        # 核心逻辑
        if self.phase == 2:
            if 'policy_net' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                logger.info("✅ Phase 2 model loaded successfully.")
            else:
                logger.error("❌ Checkpoint invalid for Phase 2 (missing 'policy_net')")

        else:
            # Phase 3 模式：兼容两种情况
            if 'q_network' in checkpoint:
                # 情况 A: 加载 RL 训练过的模型
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_q_network.load_state_dict(checkpoint['q_network'])
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("✅ Phase 3 RL model loaded (Full Resume).")

            elif 'policy_net' in checkpoint:
                # 情况 B: 加载 IL 预训练模型 (Transfer Learning)
                logger.info("⚠️ Detected IL Model (policy_net). Extracting Encoder for RL...")

                # 1. 临时实例化一个 Policy Net
                temp_policy = HierarchicalPolicy(self.cfg).to(self.device)
                temp_policy.load_state_dict(checkpoint['policy_net'])

                # 2. 提取 Encoder 权重
                encoder_state = temp_policy.gnn.encoder.state_dict()

                # 3. 注入到 Q-Network 的 Encoder
                self.q_network.encoder.load_state_dict(encoder_state)
                self.target_q_network.encoder.load_state_dict(encoder_state)

                logger.info("✅ Encoder transferred from IL to RL Agent.")
            else:
                logger.warning("❌ Unknown checkpoint format. No weights loaded.")


def create_agent_from_config(config, phase=3):
    return HRL_DQN_Agent(config, phase)


Agent = HRL_DQN_Agent