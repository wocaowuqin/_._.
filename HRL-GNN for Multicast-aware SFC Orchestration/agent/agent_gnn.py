"""
Final production-ready Agent for HIRL-SFC with GNN-based policy.
Fixed for MulticastGATWrapperVectorized compatibility.

Fixes:
1. ✅ Correctly passes 'valid_actions' and 'current_placed_dests' to model
2. ✅ Implements Gradient Accumulation for RL updates (handles variable-sized graphs)
3. ✅ Extracts tree state from node features automatically
"""

import copy
import logging
import random
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

# Removed torch_geometric imports as we handle tensors manually now

logger = logging.getLogger(__name__)


class GraphPrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self._min_priority = 1e-6
        self._max_priority = 1e6

    def add(self, state, action, reward, next_state, done, goal, next_valid_mask):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        max_prio = np.clip(max_prio, self._min_priority, self._max_priority)

        # Store experience
        data = (state, int(action), float(reward), next_state, bool(done), int(goal), next_valid_mask)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, beta: float = 0.4):
        current_len = len(self.buffer)
        if current_len == 0: return None, None, None

        priorities = self.priorities[:current_len]
        probs = np.power(priorities + 1e-8, self.alpha)
        probs /= (probs.sum() + 1e-10)

        indices = np.random.choice(current_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = np.power(current_len * probs[indices], -beta)
        weights /= (weights.max() + 1e-10)

        # Unpack
        states, actions, rewards, next_states, dones, goals, next_masks = zip(*samples)
        return (states, actions, rewards, next_states, dones, goals, next_masks), weights.astype(np.float32), indices

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            if 0 <= idx < self.capacity:
                self.priorities[idx] = np.clip(p + 1e-6, self._min_priority, self._max_priority)

    def clear(self):
        self.buffer.clear()
        self.priorities.fill(0.0)
        self.pos = 0


class Agent_SFC_GNN:
    def __init__(self, model, n_actions: int, lr: float = 1e-4, gamma: float = 0.99,
                 buffer_size: int = 10000, batch_size: int = 32, device: str = 'cuda',
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 10000,
                 prioritized_alpha: float = 0.6, prioritized_beta0: float = 0.4):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self._training = True

        self.policy_net = model.to(self.device)
        self.target_net = copy.deepcopy(model).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.memory = GraphPrioritizedReplayBuffer(buffer_size, alpha=prioritized_alpha)
        self.prioritized_beta0 = prioritized_beta0
        self.update_count = 0

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / max(1, self.epsilon_decay))

    def _extract_tree_nodes(self, x_tensor: torch.Tensor) -> List[int]:
        """
        Helper: Extract placed node indices from feature matrix.
        Assuming feature index 5 is 'is_in_tree'.
        """
        # x_tensor: [num_nodes, feat_dim]
        # Column 5 is 1.0 if node is in tree
        if x_tensor.shape[1] > 5:
            # Get indices where feature 5 is > 0.5
            indices = (x_tensor[:, 5] > 0.5).nonzero().flatten().tolist()
            return indices
        return []

    def select_action(self, state, goal_idx, valid_actions, epsilon=None, expert_action=None, beta=0.0):
        if not valid_actions: return 0

        self.steps_done += 1

        # 1. DAgger / Expert Guidance
        if expert_action is not None and expert_action in valid_actions:
            if random.random() < beta:
                return expert_action

        # 2. Epsilon-Greedy
        if epsilon is None: epsilon = self.get_epsilon()
        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        # 3. RL Policy
        x, ei, ea, req = state

        # Prepare tensors
        x_d = x.to(self.device)
        ei_d = ei.to(self.device)
        ea_d = ea.to(self.device) if ea is not None else None

        # Safe clone for request vector
        if isinstance(req, torch.Tensor):
            req_d = req.clone().detach().to(self.device)
        else:
            req_d = torch.tensor(req, dtype=torch.float32, device=self.device)

        # Ensure proper shape
        if req_d.dim() > 1: req_d = req_d.squeeze(0)

        # Extract tree nodes for model
        placed_dests = self._extract_tree_nodes(x_d)

        with torch.no_grad():
            # [FIX] Call with correct arguments for Vectorized Wrapper
            q_values = self.policy_net.forward_low(
                x=x_d,
                edge_index=ei_d,
                edge_attr=ea_d,
                req=req_d,
                goal=int(goal_idx),
                current_placed_dests=placed_dests,
                valid_actions=valid_actions  # Explicitly pass valid actions
            )

        # q_values is [1, num_valid_actions]
        q_np = q_values.cpu().numpy().flatten()

        # Argmax over valid actions
        # Note: q_values corresponds index-wise to valid_actions list
        best_idx_local = np.argmax(q_np)
        chosen = int(valid_actions[best_idx_local])

        return chosen

    def store(self, state, action, reward, next_state, done, goal, next_valid_actions):
        if next_valid_actions is None or len(next_valid_actions) == 0:
            return  # 🚫 丢弃不可信样本

        mask = np.zeros(self.n_actions, dtype=bool)
        valid = [a for a in next_valid_actions if 0 <= a < self.n_actions]
        if not valid:
            return

        mask[valid] = True

        self.memory.add(state, action, reward, next_state, done, goal, mask)

    def update(self, perform_logging=False):
        """
        RL Update using Gradient Accumulation (since model handles one graph at a time)
        """
        if self.memory.size() < self.batch_size: return 0.0

        beta = min(1.0, self.prioritized_beta0 + (self.update_count * (1.0 - self.prioritized_beta0) / 100000.0))
        batch, weights, idxes = self.memory.sample(self.batch_size, beta)
        if batch is None: return 0.0

        states, acts, rews, next_states, dones, goals, next_masks = batch

        self.optimizer.zero_grad()
        total_loss = 0.0
        td_errors = []

        # Loop through batch (Gradient Accumulation)
        for i in range(self.batch_size):
            # 1. Prepare Current State
            x, ei, ea, req = states[i]
            x_d = x.to(self.device)
            ei_d = ei.to(self.device)
            ea_d = ea.to(self.device) if ea is not None else None
            req_d = req.clone().detach().to(self.device)
            if req_d.dim() > 1: req_d = req_d.squeeze(0)

            placed_dests = self._extract_tree_nodes(x_d)
            action = int(acts[i])
            goal = int(goals[i])

            # Get Q(s, a) - we only query the chosen action to save compute
            q_pred_vec = self.policy_net.forward_low(
                x=x_d, edge_index=ei_d, edge_attr=ea_d, req=req_d,
                goal=goal, current_placed_dests=placed_dests,
                valid_actions=[action]
            )
            q_val = q_pred_vec[0, 0]  # Scalar

            # 2. Prepare Next State (Target)
            reward = float(rews[i])
            done = bool(dones[i])

            with torch.no_grad():
                if done:
                    target = reward
                else:
                    nx, nei, nea, nreq = next_states[i]
                    nx_d = nx.to(self.device)
                    nei_d = nei.to(self.device)
                    nea_d = nea.to(self.device) if nea is not None else None
                    nreq_d = nreq.clone().detach().to(self.device)
                    if nreq_d.dim() > 1: nreq_d = nreq_d.squeeze(0)

                    next_placed = self._extract_tree_nodes(nx_d)

                    # Reconstruct valid actions from mask
                    next_mask = next_masks[i]
                    valid_next_actions = np.where(next_mask)[0].tolist()

                    if not valid_next_actions:
                        target = reward
                    else:
                        # Double DQN: Select action using Online Net, Evaluate using Target Net

                        # A. Select
                        q_next_online = self.policy_net.forward_low(
                            x=nx_d, edge_index=nei_d, edge_attr=nea_d, req=nreq_d,
                            goal=goal, current_placed_dests=next_placed,
                            valid_actions=valid_next_actions
                        )
                        best_next_idx = q_next_online.argmax(dim=1).item()
                        best_next_action = valid_next_actions[best_next_idx]

                        # B. Evaluate
                        q_next_target_vec = self.target_net.forward_low(
                            x=nx_d, edge_index=nei_d, edge_attr=nea_d, req=nreq_d,
                            goal=goal, current_placed_dests=next_placed,
                            valid_actions=[best_next_action]  # Only eval best
                        )
                        q_next = q_next_target_vec[0, 0].item()

                        target = reward + self.gamma * q_next

            # 3. Loss
            target_t = torch.tensor(target, device=self.device)
            weight_t = torch.tensor(weights[i], device=self.device)

            # Hubber Loss or MSE
            loss = F.smooth_l1_loss(q_val, target_t) * weight_t

            # Backward (Accumulate)
            loss.backward()

            total_loss += loss.item()
            td_errors.append(abs(q_val.item() - target))

        # 4. Step
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update Priorities
        self.memory.update_priorities(idxes, td_errors)

        self.update_count += 1
        if self.update_count % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_loss / self.batch_size

    def supervised_update(self, batch_data: List[Dict]) -> Tuple[float, float]:
        """
        Supervised update (Imitation Learning).
        Handles Gradient Accumulation internally.
        """
        if not batch_data: return 0.0, 0.0

        self.policy_net.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        correct_predictions = 0

        for item in batch_data:
            state = item['state']
            goal = int(item['goal'])
            expert_action = int(item['action'])
            valid_actions = item['valid_actions']

            x, ei, ea, req = state
            x_d = x.to(self.device)
            ei_d = ei.to(self.device)
            ea_d = ea.to(self.device) if ea is not None else None
            req_d = req.clone().detach().to(self.device)
            if req_d.dim() > 1: req_d = req_d.squeeze(0)

            placed_dests = self._extract_tree_nodes(x_d)

            # Check if expert action is valid
            if expert_action not in valid_actions:
                valid_actions.append(expert_action)  # Ensure it's in the set

            # Forward
            # q_values will be [1, len(valid_actions)]
            q_values = self.policy_net.forward_low(
                x=x_d, edge_index=ei_d, edge_attr=ea_d, req=req_d,
                goal=goal, current_placed_dests=placed_dests,
                valid_actions=valid_actions
            )

            # Target index in the local q_values vector
            target_idx = valid_actions.index(expert_action)
            target_t = torch.tensor([target_idx], dtype=torch.long, device=self.device)

            # Loss
            loss = F.cross_entropy(q_values, target_t)
            loss.backward()
            total_loss += loss.item()

            # Accuracy
            pred_idx = q_values.argmax(dim=1).item()
            if valid_actions[pred_idx] == expert_action:
                correct_predictions += 1

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return total_loss / len(batch_data), (correct_predictions / len(batch_data)) * 100.0

    def evaluate_imitation(self, eval_data: List[Dict], num_samples: int = 500) -> Dict:
        self.policy_net.eval()
        if len(eval_data) > num_samples:
            eval_samples = random.sample(eval_data, num_samples)
        else:
            eval_samples = eval_data

        correct = 0
        total = 0

        with torch.no_grad():
            for item in eval_samples:
                state = item['state']
                goal = int(item['goal'])
                expert_action = int(item['action'])
                valid_actions = item['valid_actions']

                x, ei, ea, req = state
                x_d = x.to(self.device)
                ei_d = ei.to(self.device)
                ea_d = ea.to(self.device) if ea is not None else None
                req_d = req.clone().detach().to(self.device)
                if req_d.dim() > 1: req_d = req_d.squeeze(0)

                placed_dests = self._extract_tree_nodes(x_d)

                # Ensure expert action is valid
                if expert_action not in valid_actions:
                    continue  # Skip invalid eval samples

                q_values = self.policy_net.forward_low(
                    x=x_d, edge_index=ei_d, edge_attr=ea_d, req=req_d,
                    goal=goal, current_placed_dests=placed_dests,
                    valid_actions=valid_actions
                )

                pred_idx = q_values.argmax(dim=1).item()
                if valid_actions[pred_idx] == expert_action:
                    correct += 1
                total += 1

        self.policy_net.train()
        return {'accuracy': (correct / total * 100.0) if total > 0 else 0.0}

    def switch_to_imitation_mode(self):
        if not hasattr(self, 'rl_optimizer'):
            self.rl_optimizer = self.optimizer
        self.imitation_lr = self.lr * 5.0
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.imitation_lr)
        logger.info(f"Switched to Imitation Mode (LR={self.imitation_lr})")

    def switch_to_rl_mode(self, start_epsilon: float = 0.3):
        if hasattr(self, 'rl_optimizer'):
            self.optimizer = self.rl_optimizer
        else:
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.steps_done = 0
        self.epsilon_start = start_epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 50000
        logger.info(f"Switched to RL Mode (Start Eps={start_epsilon})")

    def save(self, path):
        torch.save({'policy_net': self.policy_net.state_dict()}, path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device)['policy_net'])
        self.target_net.load_state_dict(self.policy_net.state_dict())