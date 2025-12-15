import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """
    阶段 3：RL 微调训练器
    """

    def __init__(self, env, agent, output_dir, config):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        self.max_episodes = config.get('episodes', 2000)
        self.save_freq = config.get('save_freq', 100)

        self.stats = {'rewards': [], 'success_rate': []}

    def run(self):
        logger.info("🚀 Starting Phase 3: RL Fine-tuning...")

        # 确保加载 Phase 3 数据
        if hasattr(self.env, 'load_dataset'):
            self.env.load_dataset('phase3')

        for ep in tqdm(range(self.max_episodes), desc="RL Training"):
            try:
                # 1. 重置
                state = self.env.reset()
                req = self.env.current_request
                if req is None: break

                done = False
                ep_reward = 0

                while not done:
                    # 2. 获取 Mask
                    # 调用 PolicyHelper 获取合法的 High/Low 动作掩码
                    # (这部分需要 PolicyHelper 提供接口，这里暂用全1替代)
                    # high_mask = ...
                    # low_mask = ...
                    masks = (np.ones(self.env.NB_HIGH_LEVEL_GOALS),
                             np.ones(self.env.NB_LOW_LEVEL_ACTIONS))

                    # 3. Agent 决策
                    high_act, low_act = self.agent.select_action(state, masks)

                    # 4. 执行 High Level
                    _, r_h, _, info_h = self.env.step_high_level(high_act)

                    # 5. 执行 Low Level
                    next_state, r_l, sub_done, req_done, info_l = self.env.step_low_level(low_act)

                    # 综合奖励
                    reward = r_h + r_l

                    # 6. 存储经验 (State, Action, Reward, Next_State, Done)
                    self.agent.store_transition((state, (high_act, low_act), reward, next_state, req_done, masks))

                    # 7. 更新模型
                    loss = self.agent.update()

                    state = next_state
                    ep_reward += reward
                    done = req_done

                # 记录统计
                self.stats['rewards'].append(ep_reward)
                if ep % 10 == 0:
                    acc_rate = self.env.total_requests_accepted / max(1, self.env.total_requests_seen)
                    logger.info(f"Ep {ep} | Reward: {ep_reward:.2f} | Acc Rate: {acc_rate:.2%}")

                # 保存
                if ep % self.save_freq == 0:
                    self.agent.save(self.output_dir / f"rl_model_ep{ep}.pth")

            except Exception as e:
                logger.error(f"Phase 3 Ep {ep} Error: {e}")
                continue

        self.agent.save(self.output_dir / "rl_model_final.pth")
        logger.info("Phase 3 Complete.")
        return self.stats