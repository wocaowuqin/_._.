"""
trainer/phase3_rl_trainer.py - 修复版

主要修复:
1. ✅ 使用真实的 action masks（不再是np.ones）
2. ✅ 添加 TensorBoard 日志
3. ✅ 正确的奖励累积
4. ✅ DAgger 支持
"""

import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """
    阶段 3：RL 微调训练器（修复版）
    """

    def __init__(self, env, agent, output_dir, config):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        # 训练参数
        phase3_cfg = config.get('phase3', {})
        self.max_episodes = phase3_cfg.get('episodes', 2000)
        self.save_freq = phase3_cfg.get('save_every', 100)
        self.eval_freq = phase3_cfg.get('eval_every', 50)

        # DAgger参数
        dagger_cfg = phase3_cfg.get('dagger', {})
        self.initial_beta = dagger_cfg.get('initial_beta', 0.8)
        self.final_beta = dagger_cfg.get('final_beta', 0.05)
        self.beta_decay_steps = dagger_cfg.get('decay_steps', 3000000)
        self.global_step = 0

        # 早停参数
        early_stop_cfg = phase3_cfg.get('early_stopping', {})
        self.early_stop_enabled = early_stop_cfg.get('enabled', True)
        self.early_stop_patience = early_stop_cfg.get('patience', 50)
        self.min_improvement = early_stop_cfg.get('min_improvement', 0.01)
        self.best_acc_rate = 0.0
        self.no_improve_count = 0

        # 统计
        self.stats = {
            'rewards': [],
            'success_rate': [],
            'backup_usage': [],
            'losses': []
        }

        # 🔥 添加 TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'runs'))

        logger.info("=" * 60)
        logger.info("Phase 3 RL Trainer Initialized")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info(f"  DAgger: beta={self.initial_beta:.2f}→{self.final_beta:.2f}")
        logger.info(f"  Early Stop: {'Enabled' if self.early_stop_enabled else 'Disabled'}")
        logger.info("=" * 60)

    def get_dagger_beta(self) -> float:
        """计算当前DAgger系数"""
        if self.global_step >= self.beta_decay_steps:
            return self.final_beta

        decay_ratio = self.global_step / self.beta_decay_steps
        beta = self.initial_beta + (self.final_beta - self.initial_beta) * decay_ratio
        return beta

    def run(self):
        logger.info("🚀 Starting Phase 3: RL Fine-tuning...")

        # 确保加载 Phase 3 数据
        if hasattr(self.env, 'load_dataset'):
            success = self.env.load_dataset('phase3')
            if not success:
                logger.error("❌ Failed to load Phase 3 dataset")
                return

        for ep in tqdm(range(self.max_episodes), desc="RL Training"):
            try:
                ep_reward, ep_info = self._run_episode(ep)

                # 记录统计
                self.stats['rewards'].append(ep_reward)

                # 记录到TensorBoard
                self.writer.add_scalar('Reward/episode', ep_reward, ep)

                # 每N个episode记录详细统计
                if ep % 10 == 0:
                    acc_rate = self.env.total_requests_accepted / max(1, self.env.total_requests_seen)
                    self.stats['success_rate'].append(acc_rate)

                    backup_metrics = self.env.get_backup_metrics()
                    backup_rate = backup_metrics.get('activation_rate', 0.0) / 100.0
                    self.stats['backup_usage'].append(backup_rate)

                    # 记录到TensorBoard
                    self.writer.add_scalar('Metrics/acceptance_rate', acc_rate, ep)
                    self.writer.add_scalar('Metrics/backup_usage', backup_rate, ep)
                    self.writer.add_scalar('Metrics/epsilon', self.agent.get_epsilon(), ep)
                    self.writer.add_scalar('Metrics/dagger_beta', self.get_dagger_beta(), ep)

                    logger.info(
                        f"Ep {ep:4d} | Reward: {ep_reward:7.2f} | "
                        f"AccRate: {acc_rate:.2%} | Backup: {backup_rate:.2%} | "
                        f"Eps: {self.agent.get_epsilon():.3f} | Beta: {self.get_dagger_beta():.3f}"
                    )

                    # 早停检查
                    if self.early_stop_enabled:
                        if acc_rate > self.best_acc_rate + self.min_improvement:
                            self.best_acc_rate = acc_rate
                            self.no_improve_count = 0
                            # 保存最佳模型
                            self.agent.save(self.output_dir / "rl_model_best.pth")
                            logger.info(f"✨ New best model saved! AccRate: {acc_rate:.2%}")
                        else:
                            self.no_improve_count += 1

                        if self.no_improve_count >= self.early_stop_patience:
                            logger.info(f"🛑 Early stopping triggered at episode {ep}")
                            break

                # 定期保存
                if ep % self.save_freq == 0 and ep > 0:
                    self.agent.save(self.output_dir / f"rl_model_ep{ep}.pth")

                # 定期评估
                if ep % self.eval_freq == 0 and ep > 0:
                    eval_metrics = self._evaluate()
                    self.writer.add_scalar('Eval/acceptance_rate', eval_metrics['acc_rate'], ep)
                    logger.info(f"📊 Eval: AccRate={eval_metrics['acc_rate']:.2%}")

            except Exception as e:
                logger.error(f"Phase 3 Ep {ep} Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 训练结束，保存最终模型
        self.agent.save(self.output_dir / "rl_model_final.pth")
        self.writer.close()

        logger.info("=" * 60)
        logger.info("Phase 3 Complete")
        logger.info(f"  Best AccRate: {self.best_acc_rate:.2%}")
        logger.info(f"  Total Steps: {self.global_step}")
        logger.info("=" * 60)

        return self.stats

    def _run_episode(self, ep: int):
        """运行单个episode"""
        # 重置环境
        state = self.env.reset()
        req = self.env.current_request

        if req is None:
            return 0.0, {}

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_backup_used = 0

        beta = self.get_dagger_beta()

        while not done:
            # 🔥 获取真实的 Masks（关键修复）
            high_cands = self.env.get_expert_high_level_candidates(state, top_k=10)
            high_mask = self.env.get_high_level_candidate_mask(high_cands)
            low_mask = self.env.get_low_level_action_mask()

            masks = (high_mask, low_mask)

            # 🔥 获取专家动作（DAgger）
            expert_high = self.env.get_expert_high_level_goal(state) if high_cands else None
            expert_low = self.env.expert_low_level_action(expert_high) if expert_high is not None else None
            expert_action = (expert_high, expert_low) if expert_high is not None and expert_low >= 0 else None

            # Agent 决策
            high_act, low_act = self.agent.select_action(
                state,
                masks=masks,
                expert_action=expert_action,
                beta=beta
            )

            # 执行 High Level
            _, r_h, _, info_h = self.env.step_high_level(high_act)

            # 执行 Low Level
            next_state, r_l, sub_done, req_done, info_l = self.env.step_low_level(low_act)

            # 综合奖励
            reward = r_h + r_l

            # 记录备份使用
            if info_l.get('backup_used', False):
                ep_backup_used += 1

            # 🔥 获取下一状态的有效动作（用于存储经验）
            if not req_done:
                next_high_cands = self.env.get_expert_high_level_candidates(next_state, top_k=10)
                next_low_mask = self.env.get_low_level_action_mask()
                next_valid_actions = np.where(next_low_mask > 0.5)[0].tolist()
            else:
                next_valid_actions = []

            # 存储经验
            self.agent.store(
                state,
                (high_act, low_act),  # 完整动作
                reward,
                next_state,
                req_done,
                high_act,  # goal
                next_valid_actions
            )

            # 更新模型
            loss = self.agent.update()
            if loss is not None and loss > 0:
                self.stats['losses'].append(loss)
                # 🔥 记录Loss到TensorBoard
                self.writer.add_scalar('Loss/train', loss, self.global_step)

            # 更新状态
            state = next_state
            ep_reward += reward
            done = req_done
            ep_steps += 1
            self.global_step += 1

            # 防止episode过长
            if ep_steps > 200:
                logger.warning(f"Episode {ep} exceeded 200 steps, terminating")
                break

        info = {
            'steps': ep_steps,
            'backup_used': ep_backup_used,
            'backup_ratio': ep_backup_used / max(1, ep_steps)
        }

        return ep_reward, info

    def _evaluate(self, num_episodes: int = 10):
        """评估当前策略"""
        self.agent.eval()

        total_rewards = []
        total_accepted = 0
        total_seen = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            req = self.env.current_request

            if req is None:
                continue

            done = False
            ep_reward = 0.0

            while not done:
                # 使用贪婪策略（epsilon=0）
                high_cands = self.env.get_expert_high_level_candidates(state, top_k=10)
                high_mask = self.env.get_high_level_candidate_mask(high_cands)
                low_mask = self.env.get_low_level_action_mask()

                masks = (high_mask, low_mask)

                # 保存当前epsilon
                old_epsilon = self.agent.epsilon_start
                self.agent.epsilon_start = 0.0  # 暂时设为0（贪婪）

                high_act, low_act = self.agent.select_action(state, masks=masks)

                # 恢复epsilon
                self.agent.epsilon_start = old_epsilon

                _, r_h, _, _ = self.env.step_high_level(high_act)
                next_state, r_l, _, req_done, _ = self.env.step_low_level(low_act)

                reward = r_h + r_l
                ep_reward += reward
                state = next_state
                done = req_done

            total_rewards.append(ep_reward)

        # 收集环境统计
        total_accepted = self.env.total_requests_accepted
        total_seen = self.env.total_requests_seen
        acc_rate = total_accepted / max(1, total_seen)

        self.agent.train()  # 恢复训练模式

        return {
            'avg_reward': np.mean(total_rewards) if total_rewards else 0.0,
            'acc_rate': acc_rate
        }