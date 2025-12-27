"""
Phase 3 RL Trainer - Goal-Conditioned HRL + DAgger (修复版)
===============================================================================
修复内容：
1. 🔧 select_action 调用接口匹配 (action_mask & 返回值解包)
2. ✅ 进度条增强：新增 'Res' (资源剩余率)
3. ✅ 资源监控：实时计算网络 CPU 和 Bandwidth 的平均剩余比例。
===============================================================================
"""

import logging
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.visualizer import SFCVisualizer

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """Phase 3: Goal-Conditioned RL Trainer with DAgger & Full Metrics"""

    def __init__(self, env, agent, output_dir, config):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        # 🔥 初始化可视化器
        self.visualizer = None
        if hasattr(env, 'topo'):
            try:
                self.visualizer = SFCVisualizer(env.topo, output_dir)
                logger.info("🎨 可视化器已就绪 (plots 将保存在 outputs/checkpoints/plots)")
            except Exception as e:
                logger.warning(f"⚠️ 可视化器初始化失败: {e}")

        phase3_cfg = config.get("phase3", {})
        self.max_episodes = phase3_cfg.get("episodes", 1000)
        self.save_freq = phase3_cfg.get("save_every", 100)
        self.eval_freq = phase3_cfg.get("eval_every", 50)

        # 1. Epsilon 配置
        epsilon_cfg = phase3_cfg.get("epsilon", {})
        self.epsilon_initial = epsilon_cfg.get("initial", 0.5)
        self.epsilon_final = epsilon_cfg.get("final", 0.01)
        self.epsilon_decay_steps = epsilon_cfg.get("decay_steps", 5000)

        # 2. DAgger 配置
        dagger_cfg = phase3_cfg.get("dagger", {})
        self.use_dagger = dagger_cfg.get("enabled", True)
        self.beta = dagger_cfg.get("initial_beta", 0.8)
        self.beta_final = dagger_cfg.get("final_beta", 0.05)
        self.beta_decay_steps = dagger_cfg.get("decay_steps", 10000)

        self.min_buffer_size = 1000

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # 统计信息
        self.stats = {
            "rewards": [],
            "acceptance_rates": [],
            "blocking_rates": [],
            "resource_levels": [],  # 🔥 新增：资源剩余率记录
            "losses": [],
            "subgoal_completion_rate": [],
            "intrinsic_rewards": [],
            "total_arrived": [],
            "total_accepted": [],
            "total_completed": [],
            "total_blocked": []
        }
        self.global_step = 0

    def _calculate_epsilon(self, steps):
        """计算 Epsilon"""
        if steps >= self.epsilon_decay_steps:
            return self.epsilon_final
        decay_rate = (self.epsilon_initial - self.epsilon_final) / self.epsilon_decay_steps
        return self.epsilon_initial - (decay_rate * steps)

    def _get_network_resource_level(self):
        """🔥 计算当前网络的平均资源剩余率 (0% - 100%)"""
        try:
            rm = self.env.resource_mgr
            # 1. CPU 剩余率 (平均)
            avg_cpu = np.mean(rm.C) / np.mean(rm.C_cap)

            # 2. 带宽 剩余率 (平均)
            avg_bw = np.mean(rm.B) / np.mean(rm.B_cap)

            # 3. 综合剩余率
            return (avg_cpu + avg_bw) * 50.0  # (cpu + bw) / 2 * 100
        except:
            return 0.0

    def run(self):
        """运行训练主循环"""
        logger.info(f"🚀 Starting Training: DAgger={self.use_dagger}, Beta={self.beta}")
        self.env.set_dynamic_mode(True)
        # 显式执行一次 Hard Reset 确保起点干净
        if hasattr(self.env.resource_mgr, 'reset'):
            self.env.resource_mgr.reset(hard=True)

        self.env.reset()
        # 确保数据加载
        if hasattr(self.env, 'data_loader') and (not hasattr(self.env.data_loader, 'requests') or len(self.env.data_loader.requests) == 0):
             if hasattr(self.env, "load_dataset"):
                 self.env.load_dataset("phase3")

        pbar = tqdm(range(self.max_episodes), desc="RL Training")
        for ep in pbar:
            try:
                ep_reward, ep_info = self._run_episode(ep)

                # 1. 获取当前资源剩余量
                curr_res_level = self._get_network_resource_level()

                # 2. 更新统计列表
                self.stats["rewards"].append(ep_reward)
                self.stats["acceptance_rates"].append(ep_info["acceptance_rate"])
                self.stats["blocking_rates"].append(ep_info["blocking_rate"])
                self.stats["resource_levels"].append(curr_res_level)

                if "subgoal_completion_rate" in ep_info:
                    self.stats["subgoal_completion_rate"].append(ep_info["subgoal_completion_rate"])

                # 3. TensorBoard
                self.writer.add_scalar("Train/Reward", ep_reward, ep)
                self.writer.add_scalar("Train/AcceptanceRate", ep_info["acceptance_rate"], ep)
                self.writer.add_scalar("Train/BlockingRate", ep_info["blocking_rate"], ep)
                self.writer.add_scalar("Train/ResourceRemaining", curr_res_level, ep)
                if hasattr(self.agent, 'epsilon_low'):
                    self.writer.add_scalar("Train/Epsilon", self.agent.epsilon_low, ep)
                self.writer.add_scalar("Train/Beta", self.beta, ep)

                # 4. 计算滑动平均 (最近50轮)
                def get_avg(key):
                    data = self.stats.get(key, [])
                    recent = data[-50:]
                    return sum(recent) / len(recent) if recent else 0.0

                avg_acc = get_avg("acceptance_rates")
                avg_blk = get_avg("blocking_rates")

                # 5. 进度条更新
                expert_usage_pct = ep_info.get('expert_usage', 0) * 100
                pbar.set_postfix({
                    "Rw": f"{ep_reward:.1f}",
                    "Exp": f"{expert_usage_pct:.0f}%",
                    "Acc": f"{avg_acc:.0f}%",
                    "Blk": f"{avg_blk:.0f}%",
                    "Res": f"{curr_res_level:.1f}%"
                })

                # 保存模型
                if (ep + 1) % self.save_freq == 0:
                    self.agent.save(str(self.output_dir / f"rl_model_ep{ep+1}.pth"))

            except Exception as e:
                logger.error(f"❌ Episode {ep} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.agent.save(str(self.output_dir / "rl_model_final.pth"))
        logger.info("✅ Phase 3 Training Complete")

    def _run_episode(self, episode_idx: int):
        """
        运行一个episode（集成黑名单）

        主要修改：
        1. 从环境获取action_mask和blacklist_info
        2. 智能专家判断（检查Mask）
        3. 传入blacklist_info到Agent
        4. 记录失败节点
        5. Episode统计中添加黑名单信息
        """
        import numpy as np
        import random

        # ✅ 重置环境，获取初始mask和黑名单
        state, reset_info = self.env.reset()
        action_mask = reset_info.get('action_mask')
        blacklist_info = reset_info.get('blacklist_info', {})

        # 获取初始目标列表
        unconnected_dests = self._get_current_destinations()

        done = False
        steps = 0
        episode_reward = 0
        max_steps = self.config.get('max_steps_per_episode', 1000)

        # Episode统计
        expert_steps = 0
        agent_steps = 0
        masked_expert_steps = 0  # 被过滤的专家建议次数

        while not done and steps < max_steps:
            # ============================================
            # ✅ 智能专家判断（检查Mask）
            # ============================================
            use_expert = False
            expert_action = None

            if self.use_dagger and random.random() < self.beta:
                # 获取专家建议
                expert_suggestion = self._get_expert_action(state)

                # ✅ 检查专家建议是否被Mask
                if action_mask is None:
                    # 没有mask，直接使用
                    use_expert = True
                    expert_action = expert_suggestion
                    expert_steps += 1
                    logger.debug(f"✅ 采用专家建议: {expert_action}")
                else:
                    # 有mask，需要检查
                    valid_actions = np.where(action_mask > 0)[0]

                    if expert_suggestion in valid_actions:
                        # 专家建议有效
                        use_expert = True
                        expert_action = expert_suggestion
                        expert_steps += 1
                        logger.debug(f"✅ 采用专家建议: {expert_action}")
                    else:
                        # 专家建议被Mask
                        use_expert = False
                        masked_expert_steps += 1
                        logger.debug(
                            f"⚠️ 专家建议{expert_suggestion}被Mask "
                            f"(valid: {valid_actions[:5] if len(valid_actions) > 0 else 'none'})"
                        )

            # ============================================
            # ✅ Agent选择动作（传入黑名单信息）
            # ============================================
            high_action, low_action, action_info = self.agent.select_action(
                state=state,
                unconnected_dests=unconnected_dests,
                action_mask=action_mask,
                use_expert=use_expert,
                expert_action=expert_action,
                blacklist_info=blacklist_info  # 🚀 传入黑名单
            )

            # 统计Agent步数
            if not use_expert:
                agent_steps += 1

            # ============================================
            # 执行动作
            # ============================================
            next_state, reward, done, truncated, step_info = self.env.step(low_action)

            # ✅ 记录失败
            if not step_info.get('success', True):
                reason = step_info.get('message', 'unknown')
                if "资源不足" in reason or "访问超限" in reason:
                    self.agent.record_failure(low_action, reason)
                    logger.debug(f"📝 记录失败: 节点{low_action}, 原因:{reason}")

            # ============================================
            # 存储经验
            # ============================================
            if action_info.get('source', '').startswith('agent'):
                # High-Level经验
                if action_info.get('high_level_decision', False):
                    goal = unconnected_dests[high_action] if unconnected_dests and high_action < len(
                        unconnected_dests) else -1
                    self.agent.store_transition_high(
                        state, goal, reward, next_state, done or truncated
                    )

                # Low-Level经验
                self.agent.store_transition_low(
                    state, low_action, reward, next_state, done or truncated
                )

            # ============================================
            # ✅ 更新状态和黑名单信息
            # ============================================
            state = next_state
            action_mask = step_info.get('action_mask')
            blacklist_info = step_info.get('blacklist_info', {})

            # 更新未连接目标列表
            unconnected_dests = self._get_current_destinations()

            episode_reward += reward
            steps += 1

            # 定期更新策略
            if hasattr(self, 'update_frequency') and steps % self.update_frequency == 0:
                losses = self.agent.update_policies()

                # 记录损失
                if hasattr(self, 'writer'):
                    if losses.get('low_loss', 0) > 0:
                        self.writer.add_scalar('Loss/low', losses['low_loss'], self.global_step)
                    if losses.get('high_loss', 0) > 0:
                        self.writer.add_scalar('Loss/high', losses['high_loss'], self.global_step)

                if hasattr(self, 'global_step'):
                    self.global_step += 1

            # 检查是否因超时而终止
            if truncated:
                done = True

        # ============================================
        # ✅ Episode总结（包含黑名单统计）
        # ============================================
        blacklist_stats = self.agent.get_blacklist_learning_stats()

        # 基本统计
        logger.info(
            f"Episode {episode_idx}: "
            f"Steps={steps}, Reward={episode_reward:.2f}, "
            f"Expert={expert_steps}, Agent={agent_steps}, "
            f"MaskedExpert={masked_expert_steps}"
        )

        # 黑名单统计
        if blacklist_info.get('total', 0) > 0:
            logger.info(
                f"  📋 黑名单: {blacklist_info['total']}个节点 "
                f"{blacklist_info.get('nodes', [])[:5]}"
            )

        # Agent学习统计
        if blacklist_stats['total_failures'] > 0:
            logger.info(
                f"  📊 失败统计: {blacklist_stats['total_failures']}次, "
                f"{blacklist_stats['unique_failed_nodes']}个不同节点"
            )

            if blacklist_stats['top_failed_nodes']:
                top3 = blacklist_stats['top_failed_nodes'][:3]
                logger.info(f"  🔥 最常失败: {[(n['node'], n['count']) for n in top3]}")

        # ============================================
        # 返回Episode统计
        # ============================================
        return {
            'steps': steps,
            'reward': episode_reward,
            'success': step_info.get('request_completed', False),
            'expert_ratio': expert_steps / steps if steps > 0 else 0,
            'agent_ratio': agent_steps / steps if steps > 0 else 0,
            'masked_expert_ratio': masked_expert_steps / steps if steps > 0 else 0,
            'blacklist_size': blacklist_info.get('total', 0),
            'total_failures': blacklist_stats.get('total_failures', 0)
        }

    def _update_parameters(self):
        if hasattr(self.agent, 'update_epsilon'): self.agent.update_epsilon()
        if self.use_dagger:
            decay = (self.beta - self.beta_final) / self.beta_decay_steps
            self.beta = max(self.beta_final, self.beta - decay)

    def _empty_stats(self):
        return {
            "steps": 0, "acceptance_rate": 0.0, "blocking_rate": 0.0,
            "expert_usage": 0.0, "subgoal_completion_rate": 0.0, "avg_intrinsic_reward": 0.0,
            "completion_rate": 0.0, "total_arrived": 0, "total_accepted": 0, "total_completed": 0, "total_blocked": 0
        }

    def _log_episode_summary(self, ep, steps, reward, completed, expert_uses, trajectory):
        status = "✅ Success" if completed else "❌ Failed"
        logger.info("-" * 60)
        logger.info(f"Ep {ep} | {status} | Rw: {reward:.2f} | Exp: {expert_uses} ({expert_uses/max(1, steps)*100:.1f}%)")
        traj_str = " ".join(trajectory[-15:])
        logger.info(f"👣 ... {traj_str}")
        logger.info("-" * 60)

    def _get_current_destinations(self):
        """
        获取当前未连接的目的地列表

        Returns:
            List[int]: 未连接目的地节点ID列表
        """
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return []

        # 获取所有目的地
        all_dests = self.env.current_request.get('dest', [])

        # 获取已连接的目的地
        connected = self.env.current_tree.get('connected_dests', set())

        # 返回未连接的
        unconnected = [d for d in all_dests if d not in connected]

        return unconnected

    # ============================================
    # 辅助方法：获取专家动作
    # ============================================
    def _get_expert_action(self, state):
        """
        获取专家动作

        Args:
            state: 当前状态

        Returns:
            int: 专家建议的动作
        """
        if not hasattr(self, 'expert_policy') or self.expert_policy is None:
            # 没有专家，返回随机动作
            return random.randint(0, self.env.n - 1)

        try:
            # 调用专家策略
            action = self.expert_policy.get_action(state)
            return int(action)
        except Exception as e:
            logger.warning(f"⚠️ 专家策略出错: {e}")
            return random.randint(0, self.env.n - 1)
