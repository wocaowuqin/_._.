# trainer/phase3_rl_trainer.py (完整修复版)
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """
    Phase 3: Pure Reinforcement Learning Trainer (修复版)

    修复记录:
    1. ✅ 修复 Epsilon 更新位置（不要在 while 循环内更新）
    2. ✅ 保留所有诊断功能
    """

    def __init__(self, env, agent, output_dir, config):

        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        phase3_cfg = config.get("phase3", {})
        self.max_episodes = phase3_cfg.get("episodes", 1000)
        self.save_freq = phase3_cfg.get("save_every", 100)
        self.eval_freq = phase3_cfg.get("eval_every", 50)

        # Epsilon 配置
        epsilon_cfg = phase3_cfg.get("epsilon", {})
        self.epsilon_initial = epsilon_cfg.get("initial", 0.5)
        self.epsilon_final = epsilon_cfg.get("final", 0.05)
        self.epsilon_decay_steps = epsilon_cfg.get("decay_steps", 50000)

        # RL 配置
        rl_cfg = phase3_cfg.get("rl", {})
        buffer_cfg = rl_cfg.get("replay_buffer", {})
        self.min_buffer_size = buffer_cfg.get("min_size", 1000)
        self.batch_size = buffer_cfg.get("batch_size", 64)

        self.global_step = 0

        # 详细统计
        self.stats = {
            "rewards": [],
            "success_rate": [],
            "losses": [],
            "epsilon_history": [],
            "step_rewards": [],
            "episode_lengths": [],
        }

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # 诊断输出
        logger.info("=" * 60)
        logger.info("Phase 3 RL Trainer (PURE RL, NO EXPERT)")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info(f"  Epsilon: {self.epsilon_initial} → {self.epsilon_final}")
        logger.info(f"  Decay Steps: {self.epsilon_decay_steps}")
        logger.info(f"  Min Buffer Size: {self.min_buffer_size}")
        logger.info(f"  Batch Size: {self.batch_size}")

        # 计算预期 Epsilon
        ep100_eps = self._calculate_epsilon(100 * 120)
        ep500_eps = self._calculate_epsilon(500 * 120)
        logger.info(f"  预计 Ep 100 Epsilon: {ep100_eps:.4f}")
        logger.info(f"  预计 Ep 500 Epsilon: {ep500_eps:.4f}")
        logger.info("=" * 60)

    def _calculate_epsilon(self, steps):
        """计算给定步数的 Epsilon 值"""
        return self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
            np.exp(-steps / self.epsilon_decay_steps)

    def _update_epsilon(self):
        """更新 Agent 的 Epsilon"""
        epsilon = self._calculate_epsilon(self.global_step)
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = epsilon
        elif hasattr(self.agent, 'set_epsilon'):
            self.agent.set_epsilon(epsilon)
        return epsilon

    def run(self):
        logger.info("🚀 Starting Phase 3: Pure RL Training")

        # 尝试加载数据
        if hasattr(self.env, "load_dataset"):
            self.env.load_dataset("phase3")

        for ep in tqdm(range(self.max_episodes), desc="RL Training"):
            try:
                ep_reward, ep_info = self._run_episode(ep)

                # 记录统计
                self.stats["rewards"].append(ep_reward)
                self.stats["episode_lengths"].append(ep_info.get("steps", 0))

                # TensorBoard 记录
                self.writer.add_scalar("Reward/episode", ep_reward, ep)
                self.writer.add_scalar("Metrics/episode_length", ep_info.get("steps", 0), ep)

                # 每 10 个 Episode 输出详细信息
                if ep % 10 == 0:
                    acc_rate = (
                            self.env.total_requests_accepted
                            / max(1, self.env.total_requests_seen)
                    )
                    self.stats["success_rate"].append(acc_rate)

                    current_epsilon = self.agent.get_epsilon() if hasattr(self.agent,
                                                                          'get_epsilon') else self.agent.epsilon

                    # 计算最近 10 个 Episode 的平均奖励
                    recent_rewards = self.stats["rewards"][-10:]
                    avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                    # 计算平均 Loss
                    recent_losses = self.stats["losses"][-100:]
                    avg_loss = np.mean(recent_losses) if recent_losses else 0.0

                    self.writer.add_scalar("Metrics/acceptance_rate", acc_rate, ep)
                    self.writer.add_scalar("Metrics/epsilon", current_epsilon, ep)
                    self.writer.add_scalar("Metrics/avg_recent_reward", avg_recent_reward, ep)
                    self.writer.add_scalar("Metrics/avg_loss", avg_loss, ep)

                    # 详细日志
                    logger.info(
                        f"Ep {ep:4d} | "
                        f"Reward {ep_reward:7.2f} (Avg10: {avg_recent_reward:7.2f}) | "
                        f"AccRate {acc_rate:.2%} | "
                        f"Eps {current_epsilon:.3f} | "
                        f"Loss {avg_loss:.4f} | "
                        f"Steps {ep_info.get('steps', 0):3d} | "
                        f"BufferSize {len(self.agent.memory) if hasattr(self.agent, 'memory') else 'N/A'}"
                    )

                # 定期保存
                if ep % self.save_freq == 0 and ep > 0:
                    self.agent.save(self.output_dir / f"rl_model_ep{ep}.pth")

                # 定期评估
                if ep % self.eval_freq == 0 and ep > 0:
                    eval_info = self._evaluate()
                    self.writer.add_scalar("Eval/acceptance_rate", eval_info["acc_rate"], ep)
                    logger.info(
                        f"  📊 Eval | AvgReward {eval_info['avg_reward']:.2f} | AccRate {eval_info['acc_rate']:.2%}")

            except Exception as e:
                logger.error(f"❌ Phase3 Episode {ep} failed: {e}")
                import traceback
                traceback.print_exc()

        # 最终保存
        self.agent.save(self.output_dir / "rl_model_final.pth")
        self.writer.close()

        # 最终统计
        logger.info("=" * 60)
        logger.info("✅ Phase 3 Training Complete")
        logger.info(f"  Total Episodes: {len(self.stats['rewards'])}")
        logger.info(f"  Final Avg Reward: {np.mean(self.stats['rewards'][-100:]):.2f}")
        logger.info(f"  Final AccRate: {self.stats['success_rate'][-1]:.2%}" if self.stats['success_rate'] else "  N/A")
        logger.info("=" * 60)

        return self.stats

    def _run_episode(self, ep: int):
        """运行单个 Episode（最终清晰版：标记任务切换）"""

        # 更新 Epsilon
        current_epsilon = self._update_epsilon()

        # 环境重置
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        # 获取首个请求信息
        req = self.env.current_request
        src_node = req.get('source', 'Err') if req else 'Err'
        dst_nodes = req.get('dest', []) if req else []

        # 🔥 轨迹记录器：增加 Request ID 标记
        req_id = req.get('id', 0) if req else 0
        real_trajectory = [f"R{req_id}:Start({src_node})"]

        if req is None: return 0.0, {"steps": 0}

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Mask & Action
            high_mask = self.env.get_high_level_action_mask()
            low_mask = self.env.get_low_level_action_mask()
            masks = (high_mask, low_mask)

            high_act, low_act = self.agent.select_action(state, masks=masks)

            # Step
            _, r_h, _, _ = self.env.step_high_level(high_act)
            step_result = self.env.step_low_level(low_act)

            if len(step_result) == 5:
                next_state, r_l, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, r_l, done, info = step_result

            # ====================================================
            # 🔥 日志逻辑优化
            # ====================================================
            if info and not info.get('error'):
                act_type = info.get('action_type', 'unknown')

                if act_type == 'move':
                    target = info.get('to', int(low_act))
                    real_trajectory.append(f"→{target}")

                elif act_type == 'deploy':
                    node = info.get('node', int(low_act))
                    real_trajectory.append(f"★Dep({node})")

                    # 🔥 检查是否完成了当前请求
                    if info.get('all_deployed'):
                        real_trajectory.append("✅Done")

                        # 如果还有下一个请求，标记新起点
                        if not done:
                            next_req = self.env.current_request
                            if next_req:
                                new_src = next_req.get('source')
                                new_id = next_req.get('id', '?')
                                real_trajectory.append(f" || 🆕 R{new_id}:Start({new_src})")

            # Store & Update
            next_high_mask = self.env.get_high_level_action_mask()
            next_low_mask = self.env.get_low_level_action_mask()

            self.agent.store_transition(
                state, (high_act, low_act), r_l, next_state, done,
                next_valid_mask=(next_high_mask, next_low_mask)
            )

            buffer_size = len(self.agent.memory) if hasattr(self.agent, 'memory') else 0
            if buffer_size >= self.min_buffer_size:
                loss = self.agent.update()
                # (Optional) Log loss...

            state = next_state
            ep_reward += r_l
            ep_steps += 1
            self.global_step += 1

            if ep_steps > 100: break

        # ====================================================
        # 🔥 打印最终清晰路径
        # ====================================================
        # 只要跑完了(无论成功失败)，只要步数大于1就打印看看
        if ep < 3 or (ep % 10 == 0):
            # 统计总共完成了多少个请求
            completed_reqs = path_str = "".join(real_trajectory).count("✅Done")

            logger.info("-" * 60)
            logger.info(f"Ep {ep} Summary | Steps: {ep_steps} | Reqs Completed: {completed_reqs}")

            # 分行打印路径，太长了很难看
            # 将路径按请求分割打印
            full_str = " ".join(real_trajectory)
            segments = full_str.split("||")

            logger.info("👣 轨迹详情:")
            for seg in segments:
                logger.info(f"   {seg.strip()}")

            logger.info(f"💰 总奖励: {ep_reward:.2f}")
            logger.info("-" * 60)

        return ep_reward, {"steps": ep_steps}
    def _evaluate(self, num_episodes: int = 5):
        """评估模型性能"""
        self.agent.eval()
        rewards = []

        for _ in range(num_episodes):
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            if self.env.current_request is None:
                continue

            done = False
            ep_reward = 0.0

            # 临时关闭 epsilon
            old_eps = self.agent.epsilon if hasattr(self.agent, 'epsilon') else 0.0
            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = 0.0

            while not done:
                high_mask = self.env.get_high_level_action_mask()
                low_mask = self.env.get_low_level_action_mask()

                high_act, low_act = self.agent.select_action(state, masks=(high_mask, low_mask))

                self.env.step_high_level(high_act)
                step_result = self.env.step_low_level(low_act)

                if len(step_result) == 5:
                    next_state, r_l, term, trunc, _ = step_result
                    done = term or trunc
                else:
                    next_state, r_l, done, _ = step_result

                ep_reward += r_l
                state = next_state

            rewards.append(ep_reward)

            # 恢复 epsilon
            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = old_eps

        self.agent.train()
        acc_rate = self.env.total_requests_accepted / max(1, self.env.total_requests_seen)
        return {"avg_reward": np.mean(rewards), "acc_rate": acc_rate}