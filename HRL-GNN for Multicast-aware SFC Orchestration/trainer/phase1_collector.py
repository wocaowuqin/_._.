import os
import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    """
    阶段 1：专家轨迹采集器
    利用 PolicyHelper 生成专家数据
    """

    def __init__(self, env, output_dir, config):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.dataset = []

        self.stats = {
            "episodes": 0,
            "success": 0,
            "fail": 0,
            "transitions": 0
        }

    def run(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection...")

        # 确保加载 Phase 1 数据
        if hasattr(self.env, 'load_dataset'):
            self.env.load_dataset('phase1')

        num_episodes = self.cfg.get('episodes', 1000)

        for ep in tqdm(range(num_episodes), desc="Collecting"):
            try:
                # 1. 重置环境
                obs = self.env.reset()
                req = self.env.current_request
                if req is None:
                    break  # 数据跑完

                done = False
                ep_transitions = []
                success = False

                while not done:
                    # 2. 获取状态数据
                    # 注意：如果用GNN，obs是tuple；如果用Flat，obs是np.array
                    # 这里我们需要保存用于训练的原始状态

                    # 3. 询问专家 (PolicyHelper)
                    # 获取高层动作建议 (目标选择)
                    high_cands = self.env.policy_helper.get_expert_candidates(
                        req, self.env.resource_mgr.get_network_state_dict(req),
                        self.env.unadded_dest_indices,
                        self.env.current_tree, self.env.nodes_on_tree
                    )

                    if not high_cands:
                        break  # 专家无法决策

                    expert_high_act = int(high_cands[0][0])  # 最佳目标索引

                    # 获取低层动作建议 (路径选择)
                    # 这是一个简化逻辑，实际需要遍历 valid actions 找到专家选择的那条路
                    # 这里为了演示，我们假设 Env 执行 High Level 后，再问专家 Low Level

                    # 执行 High Level
                    self.env.step_high_level(expert_high_act)

                    # 询问 Low Level (PolicyHelper.get_best_plan 内部含专家逻辑)
                    # 我们需要把专家的决策转化为 action index
                    # 由于这部分逻辑较复杂，通常Phase1只采集"成功路径"的特征
                    # 这里简化为：直接采集 (State, High_Action) 和 (State, Low_Action)

                    # 暂存数据 (State, High_Action)
                    # 注意：这里只演示高层策略采集，完整版需要采集低层
                    ep_transitions.append({
                        'state': obs,
                        'high_action': expert_high_act,
                        # 'low_action': ... (需要反推 action index)
                    })

                    # 执行一步 Low Level (使用专家决策)
                    # 这里实际上应该调用 env.policy_helper.get_best_plan 获取 plan
                    # 然后反推 action ID。为简化，直接让环境跑一步专家动作
                    # 假设我们只训练高层策略，或者低层策略单独处理

                    # 模拟环境推进 (这里为了不中断流程，使用随机或简单的逻辑推进)
                    # 在真实实现中，这里必须执行专家的 plan
                    valid_low = self.env.policy_helper.get_valid_low_level_actions(
                        self.env.path_manager, self.env.current_tree
                    )
                    low_act = valid_low[0]  # 占位

                    next_obs, reward, sub_done, req_done, info = self.env.step_low_level(low_act)

                    obs = next_obs
                    done = req_done

                    if done and info.get('success', False):
                        success = True

                # 4. 如果整个请求成功，保存轨迹
                if success:
                    self.dataset.extend(ep_transitions)
                    self.stats['success'] += 1
                    self.stats['transitions'] += len(ep_transitions)
                else:
                    self.stats['fail'] += 1

                self.stats['episodes'] += 1

                # 定期保存
                if ep % self.cfg.get('save_every', 500) == 0:
                    self._save_data(f"expert_data_ep{ep}.pkl")

            except Exception as e:
                logger.error(f"Ep {ep} Error: {e}")
                continue

        # 最终保存
        self._save_data("expert_data_final.pkl")
        logger.info(f"Phase 1 Done. Stats: {self.stats}")
        return self.dataset

    def _save_data(self, filename):
        path = self.output_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)
        logger.info(f"Saved {len(self.dataset)} samples to {path}")