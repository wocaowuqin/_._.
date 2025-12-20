# trainer/phase3_rl_trainer.py
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Phase3RLTrainer:
    """
    Phase 3: Pure Reinforcement Learning Trainer
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

        self.global_step = 0

        self.stats = {
            "rewards": [],
            "success_rate": [],
            "losses": [],
        }

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        logger.info("=" * 60)
        logger.info("Phase 3 RL Trainer (PURE RL, NO EXPERT)")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info("=" * 60)

    def run(self):
        logger.info("🚀 Starting Phase 3: Pure RL Training")
        
        # 尝试加载数据 (防止 Env 没有自动加载)
        if hasattr(self.env, "load_dataset"):
            self.env.load_dataset("phase3")

        for ep in tqdm(range(self.max_episodes), desc="RL Training"):
            try:
                ep_reward, ep_info = self._run_episode(ep)

                self.stats["rewards"].append(ep_reward)
                self.writer.add_scalar("Reward/episode", ep_reward, ep)

                if ep % 10 == 0:
                    acc_rate = (
                        self.env.total_requests_accepted
                        / max(1, self.env.total_requests_seen)
                    )
                    self.stats["success_rate"].append(acc_rate)

                    self.writer.add_scalar("Metrics/acceptance_rate", acc_rate, ep)
                    self.writer.add_scalar("Metrics/epsilon", self.agent.get_epsilon(), ep)

                    logger.info(
                        f"Ep {ep:4d} | Reward {ep_reward:7.2f} | "
                        f"AccRate {acc_rate:.2%} | "
                        f"Eps {self.agent.get_epsilon():.3f}"
                    )

                if ep % self.save_freq == 0 and ep > 0:
                    self.agent.save(self.output_dir / f"rl_model_ep{ep}.pth")

                if ep % self.eval_freq == 0 and ep > 0:
                    eval_info = self._evaluate()
                    self.writer.add_scalar("Eval/acceptance_rate", eval_info["acc_rate"], ep)

            except Exception as e:
                logger.error(f"Phase3 Episode {ep} failed: {e}")
                import traceback
                traceback.print_exc()

        self.agent.save(self.output_dir / "rl_model_final.pth")
        self.writer.close()
        logger.info("✅ Phase 3 Training Complete")
        return self.stats

    def _run_episode(self, ep: int):
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        if self.env.current_request is None:
            return 0.0, {}

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # 1. 获取当前 Mask
            high_mask = self.env.get_high_level_action_mask()
            low_mask = self.env.get_low_level_action_mask()
            masks = (high_mask, low_mask)

            # 2. Agent 选择动作
            high_act, low_act = self.agent.select_action(state, masks=masks)

            # 3. 环境执行
            _, r_h, _, _ = self.env.step_high_level(high_act)
            step_result = self.env.step_low_level(low_act)

            if len(step_result) == 5:
                next_state, r_l, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, r_l, done, _ = step_result

            reward = r_h + r_l

            # 4. 🔥 获取下一时刻 Mask (用于存储)
            next_high_mask = self.env.get_high_level_action_mask()
            next_low_mask = self.env.get_low_level_action_mask()
            next_masks = (next_high_mask, next_low_mask)

            # 5. 存储经验 (必须传入 next_valid_mask)
            self.agent.store_transition(
                state,
                (high_act, low_act),
                reward,
                next_state,
                done,
                next_valid_mask=next_masks # ✅ 关键修复
            )

            # 6. 更新网络
            loss = self.agent.update()
            if loss is not None and loss != 0.0:
                self.stats["losses"].append(loss)
                self.writer.add_scalar("Loss/train", loss, self.global_step)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            self.global_step += 1

            if ep_steps > 200:
                break

        return ep_reward, {"steps": ep_steps}

    def _evaluate(self, num_episodes: int = 5):
        self.agent.eval()
        rewards = []

        for _ in range(num_episodes):
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            if self.env.current_request is None: continue

            done = False
            ep_reward = 0.0
            
            # 临时关闭 epsilon
            old_eps = self.agent.epsilon
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
                
                ep_reward += r_l # 简化评估只看低层奖励或总奖励
                state = next_state

            rewards.append(ep_reward)
            self.agent.epsilon = old_eps

        self.agent.train()
        acc_rate = self.env.total_requests_accepted / max(1, self.env.total_requests_seen)
        return {"avg_reward": np.mean(rewards), "acc_rate": acc_rate}