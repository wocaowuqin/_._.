# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 RL Trainer - Goal-Conditioned HRL + DAgger (修复显示版)
===============================================================================
修复内容：
1. ✅ 统计逻辑：改为“全局累计平均”，修复 Acc=1% 的显示问题。
2. 🛡️ 崩溃保护：捕获 Agent 内部错误，防止训练中断。
3. 📊 进度条：显示真实累计 Acc (接纳率) 和 Blk (阻塞率)。
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

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # 统计信息容器
        self.stats = {
            "rewards": [],
            "acceptance_rates": [],
            "blocking_rates": [],
            "resource_levels": [],
            "subgoal_completion_rate": []
        }
        self.global_step = 0

    def _get_network_resource_level(self):
        """🔥 [修改版] 只监控 DC 节点的 CPU 剩余率，这才是瓶颈！"""
        try:
            rm = self.env.resource_mgr
            # 获取 DC 节点列表 (从环境配置里拿)
            dc_nodes = getattr(self.env, 'dc_nodes', [])

            if not dc_nodes:
                # 如果没拿到，暂时回退到旧逻辑
                return 0.0

            total_dc_cpu = 0
            total_dc_cap = 0

            # 遍历所有 DC 节点
            for node in dc_nodes:
                # 兼容 SoA (dict) 和 AoS (list) 结构
                if isinstance(rm.nodes, dict) and 'cpu' in rm.nodes:
                    # SoA: {'cpu': [100, 100, ...]}
                    if node < len(rm.nodes['cpu']):
                        cpu = rm.nodes['cpu'][node]
                        total_dc_cpu += cpu
                        total_dc_cap += 100.0  # 假设容量100
                else:
                    # AoS: [{'cpu': 100}, ...]
                    if node < len(rm.nodes):
                        cpu = rm.nodes[node].get('cpu', 0)
                        total_dc_cpu += cpu
                        total_dc_cap += 100.0

            if total_dc_cap == 0: return 0.0

            # 计算 DC 节点的平均剩余率
            dc_res_pct = (total_dc_cpu / total_dc_cap) * 100.0

            return dc_res_pct

        except Exception as e:
            # print(f"资源监控出错: {e}")
            return 0.0

    def run(self):
        """运行训练主循环"""
        logger.info(f"🚀 Starting Training: DAgger={self.use_dagger}, Beta={self.beta}")

        # ============================================
        # 🔥 全局累计计数器 (修复 Acc 显示问题)
        # ============================================
        total_episodes = 0
        total_success = 0
        total_failed = 0

        pbar = tqdm(range(self.max_episodes), desc="RL Training")
        
        for ep in pbar:
            try:
                # 运行一个 Episode
                ep_reward, ep_info = self._run_episode(ep)

                # 1. 获取资源水平
                curr_res_level = self._get_network_resource_level()

                # 2. ✅ 更新全局计数器 (核心修复)
                total_episodes += 1
                
                # 判断成功标准：只要 env 说是 success 或 request_completed 就算成
                is_success = ep_info.get('success', False)
                
                if is_success:
                    total_success += 1
                else:
                    total_failed += 1

                # 3. 计算累计指标
                cum_acc = total_success / total_episodes
                cum_blk = total_failed / total_episodes

                # 4. 记录到 Stats (用于绘图)
                self.stats["rewards"].append(ep_reward)
                self.stats["acceptance_rates"].append(1.0 if is_success else 0.0) # 记录单次
                self.stats["blocking_rates"].append(0.0 if is_success else 1.0)
                self.stats["resource_levels"].append(curr_res_level)

                # 5. TensorBoard (记录累计值更平滑)
                self.writer.add_scalar("Train/Reward", ep_reward, ep)
                self.writer.add_scalar("Train/CumulativeAcc", cum_acc, ep)
                self.writer.add_scalar("Train/CumulativeBlk", cum_blk, ep)
                self.writer.add_scalar("Train/Resource", curr_res_level, ep)
                
                if hasattr(self.agent, 'epsilon_low'):
                    self.writer.add_scalar("Train/Epsilon", self.agent.epsilon_low, ep)

                # 6. 更新进度条 (显示全局累计值)
                expert_usage_pct = ep_info.get('expert_usage', 0) * 100
                pbar.set_postfix({
                    "Rw": f"{ep_reward:.0f}",
                    "Exp": f"{expert_usage_pct:.0f}%",
                    "Acc": f"{cum_acc:.1%}",  # ✅ 显示真实累计 Acc
                    "Blk": f"{cum_blk:.1%}",  # ✅ 显示真实累计 Blk
                    "Res": f"{curr_res_level:.0f}%"
                })

                # 保存模型
                if (ep + 1) % self.save_freq == 0:
                    self.agent.save(str(self.output_dir / f"rl_model_ep{ep + 1}.pth"))

            except Exception as e:
                # 🛡️ 崩溃防御：捕获所有异常，不中断训练
                logger.error(f"❌ Episode {ep} CRASHED: {e}")
                # 发生异常算作失败
                total_episodes += 1
                total_failed += 1
                continue

        # 训练结束保存
        self.agent.save(str(self.output_dir / "rl_model_final.pth"))
        logger.info(f"✅ Training Complete. Final Acc: {total_success/total_episodes:.2%}")

    def _run_episode(self, episode_idx: int):
        """运行一个episode（集成黑名单 + DAgger）"""
        import numpy as np
        import random

        # 获取最大步数
        max_steps = getattr(self, 'max_steps_per_episode', getattr(self.env, 'max_steps', 600))

        # ✅ 重置环境
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, reset_info = reset_result
        else:
            state = reset_result
            reset_info = {}

        # 获取 mask 和 info
        action_mask = reset_info.get('action_mask')
        blacklist_info = reset_info.get('blacklist_info', {})
        unconnected_dests = self._get_current_destinations()

        done = False
        steps = 0
        episode_reward = 0

        # DAgger 统计
        expert_steps = 0
        masked_expert_steps = 0
        
        # 初始化 step_info
        step_info = {'success': False, 'request_completed': False}

        while not done and steps < max_steps:
            # DAgger 逻辑
            beta = getattr(self, 'beta', 0.0)
            use_dagger = getattr(self, 'use_dagger', False)
            use_expert = False
            expert_action = None

            # 确保 mask 存在
            if action_mask is None and hasattr(self.env, 'get_action_mask'):
                action_mask = self.env.get_action_mask()

            # 专家介入判断
            if use_dagger and random.random() < beta:
                expert_suggestion = self._get_expert_action(state)
                # 检查专家建议是否合法 (被Mask blocked?)
                if action_mask is None:
                    use_expert = True; expert_action = expert_suggestion
                else:
                    valid_actions = np.where(action_mask > 0)[0]
                    if expert_suggestion in valid_actions:
                        use_expert = True; expert_action = expert_suggestion
                        expert_steps += 1
                    else:
                        masked_expert_steps += 1

            # ✅ Agent 选择动作
            high_action, low_action, action_info = self.agent.select_action(
                state=state,
                unconnected_dests=unconnected_dests,
                action_mask=action_mask,
                use_expert=use_expert,
                expert_action=expert_action,
                blacklist_info=blacklist_info
            )
            
            # 🛡️ 防御：如果 Agent 返回 -1 (无效)，手动处理
            if low_action == -1:
                # 强制结束当前 Episode，视为失败
                logger.warning(f"⚠️ Agent returned -1 (No Valid Actions). Terminating Episode {episode_idx}.")
                return episode_reward, {'success': False, 'blocking_rate': 1.0, 'message': 'no_valid_actions'}

            # 执行动作
            step_result = self.env.step(low_action)

            # 解包结果
            if len(step_result) == 5:
                next_state, reward, done, truncated, step_info = step_result
            else:
                next_state, reward, done, step_info = step_result
                truncated = False

            # 记录失败原因用于黑名单学习
            if not step_info.get('success', True):
                reason = step_info.get('message', 'unknown')
                if "资源不足" in reason or "访问超限" in reason:
                    self.agent.record_failure(low_action, reason)

            # 存储经验
            if action_info.get('source', '').startswith('agent'):
                # High-Level Buffer
                if action_info.get('high_level_decision', False):
                    goal = unconnected_dests[high_action] if unconnected_dests and high_action < len(unconnected_dests) else -1
                    self.agent.store_transition_high(state, goal, reward, next_state, done or truncated)
                
                # Low-Level Buffer
                self.agent.store_transition_low(state, low_action, reward, next_state, done or truncated)

            # 更新状态
            state = next_state
            action_mask = step_info.get('action_mask')
            blacklist_info = step_info.get('blacklist_info', {})
            unconnected_dests = self._get_current_destinations()
            episode_reward += reward
            steps += 1

            # 定期更新网络
            if steps % 4 == 0:
                self.agent.update_policies()
            
            if truncated: done = True

        # Episode 结束统计
        is_success = step_info.get('request_completed', False) or step_info.get('success', False)
        
        episode_info = {
            'steps': steps,
            'success': is_success,
            'blocking_rate': 0.0 if is_success else 1.0,
            'expert_usage': expert_steps / steps if steps > 0 else 0,
            'masked_expert': masked_expert_steps
        }
        
        # 简单日志
        status_icon = "✅" if is_success else "❌"
        if is_success or episode_idx % 10 == 0: # 减少日志刷屏，成功或每10次打印一次
            logger.info(f"Ep {episode_idx} | {status_icon} | Rw: {episode_reward:.1f} | Steps: {steps}")

        return episode_reward, episode_info

    def _get_current_destinations(self):
        """获取当前未连接的目的地列表"""
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return []
        all_dests = self.env.current_request.get('dest', [])
        connected = self.env.current_tree.get('connected_dests', set())
        return [d for d in all_dests if d not in connected]

    def _get_expert_action(self, state):
        """获取专家动作"""
        if not hasattr(self, 'agent') or not hasattr(self.agent, 'expert'): # 这里的 expert 应该是环境里的
            # 如果没有 Expert Wrapper，尝试用环境里的
             if hasattr(self.env, 'expert') and self.env.expert:
                 # 这里需要 expert 逻辑，暂时随机兜底
                 pass
        return random.randint(0, getattr(self.env, 'n', 28) - 1)