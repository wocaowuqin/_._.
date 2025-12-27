# reward_critic.py
"""
Reward Critic Module for Multicast VNF Mapping (终极扩展版)

功能特性:
1. ✅ 统一管理 VNF 部署与树构建的所有奖励参数
2. ✅ 支持指数级连接奖励 (50 * 1.5^n)
3. ✅ 支持距离引导奖励 (Distance Guidance)
4. ✅ 兼容原有的保存/加载/诊断接口
"""

import logging
from typing import Dict, Optional, Tuple, Any, Union, List
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class RewardCriticParams:
    """奖励函数参数配置 - 扩展版"""

    # ========================================
    # 1. VNF 部署阶段 (Phase A)
    # ========================================
    vnf_deploy_success: float = 40.0  # 单个 VNF 部署成功
    vnf_all_complete: float = 50.0  # 所有 VNF 部署完成
    vnf_deploy_failed: float = -30.0  # 部署失败（资源不足）

    move_to_dc_bonus: float = 5.0  # 移动到 DC 节点
    move_cost: float = -0.1  # 基础移动成本

    # ========================================
    # 2. 树构建阶段 (Phase B)
    # ========================================
    # 连接奖励公式: base * (exponential ^ (n-1)) + progress_bonus * ratio
    connection_base: float = 60.0  # 基础连接奖励
    connection_exponential: float = 1.5  # 指数增长系数 (鼓励连得越多奖励越大)
    connection_progress_bonus: float = 50.0  # 进度奖励基数

    dest_reached_bonus: float = 30.0  # 仅仅到达目的节点（未连接）
    full_completion_bonus: float = 300.0  # 任务完美完成大奖

    # ========================================
    # 3. 导航与引导 (Guidance)
    # ========================================
    guidance_closer_rate: float = 5.0  # 向目标靠近一步
    guidance_farther_penalty: float = -2.0  # 远离目标一步
    guidance_idle_penalty: float = -0.5  # 原地不动/平移

    # ========================================
    # 4. 惩罚项 (Penalties)
    # ========================================
    invalid_link: float = -10.0  # 非法移动/断路
    invalid_action: float = -15.0  # 被 Mask 禁止的动作
    wrong_position: float = -5.0  # 在错误位置尝试操作

    # 频次惩罚 (防止死循环)
    freq_penalty_threshold: int = 5  # 阈值
    freq_penalty_rate: float = -2.0  # 惩罚系数

    # ========================================
    # 5. 超时处理 (Timeout)
    # ========================================
    timeout_high_progress_threshold: float = 0.8  # 高进度门槛
    timeout_bonus_rate: float = 50.0  # 高进度结算奖励
    timeout_penalty_rate: float = 50.0  # 低进度惩罚
    vnf_timeout_penalty: float = -50.0  # 第一阶段超时惩罚

    # ========================================
    # 6. 旧版兼容 / 归一化 (保留以防报错)
    # ========================================
    reward_scale: float = 1.0  # 全局缩放 (默认1.0表示不缩放)
    w_cpu: float = 0.33
    w_bw: float = 0.34
    w_hop: float = 0.33


@dataclass
class CriticDiagnostics:
    """诊断信息数据结构"""
    total_rewards: List[float]
    avg_reward: float
    request_count: int
    success_rate: float


class RewardCritic:
    """
    扩展版 RewardCritic - 支持 VNF 部署 + 树构建完整流程
    """

    def __init__(self,
                 training_phase: int = 3,
                 params: Optional[Union[Dict, RewardCriticParams]] = None):

        self.phase = int(training_phase)

        # 参数初始化
        if params is None:
            self.params = RewardCriticParams()
        elif isinstance(params, dict):
            # 过滤掉不匹配的键，防止报错
            valid_keys = RewardCriticParams.__annotations__.keys()
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            self.params = RewardCriticParams(**filtered_params)
        else:
            self.params = params

        # 运行时状态
        self._buffer_reward = 0.0
        self._request_active = False

        # 统计历史
        self.reward_history: List[float] = []
        self.history_len = 1000  # 限制历史长度防止内存泄漏

    # =========================================================================
    # VNF 部署阶段奖励计算
    # =========================================================================
    def compute_vnf_deploy_reward(self,
                                  success: bool,
                                  all_complete: bool = False,
                                  quality_score: float = 0.0) -> float:
        """
        计算 VNF 部署动作的奖励
        :param success: 是否部署成功
        :param all_complete: 是否所有 VNF 都部署完毕
        :param quality_score: 部署位置质量加成 (例如基于 AvgHops)
        """
        p = self.params
        if success:
            reward = p.vnf_deploy_success + quality_score
            if all_complete:
                reward += p.vnf_all_complete
            return reward * p.reward_scale
        else:
            return p.vnf_deploy_failed * p.reward_scale

    def compute_vnf_move_reward(self,
                                to_dc: bool = False,
                                valid_link: bool = True,
                                guidance_val: float = 0.0) -> float:
        """
        计算 VNF 阶段移动奖励
        :param to_dc: 是否移动到了 DC 节点
        :param valid_link: 是否是有效链路
        :param guidance_val: 额外的引导分 (由 Env 计算距离产生)
        """
        p = self.params
        if not valid_link:
            return p.invalid_link * p.reward_scale

        reward = p.move_cost + guidance_val
        if to_dc:
            reward += p.move_to_dc_bonus

        return reward * p.reward_scale

    # =========================================================================
    # 树构建阶段奖励计算
    # =========================================================================
    def compute_tree_connection_reward(self,
                                       connected_count: int,
                                       total_dests: int,
                                       is_complete: bool = False) -> float:
        """
        计算连接动作奖励 (指数递增)
        :param connected_count: 当前已连接节点数 (含本次)
        :param total_dests: 总目的节点数
        :param is_complete: 是否全部连接完成
        """
        p = self.params

        # 1. 基础指数奖励: 60 * 1.5^(n-1)
        # n=1: 60, n=2: 90, n=3: 135, n=4: 202, n=5: 303
        base = p.connection_base * (p.connection_exponential ** (connected_count - 1))

        # 2. 进度奖励 (线性)
        ratio = connected_count / total_dests if total_dests > 0 else 0
        progress = p.connection_progress_bonus * ratio

        reward = base + progress

        # 3. 完美通关大奖
        if is_complete:
            reward += p.full_completion_bonus

        return reward * p.reward_scale

    def compute_tree_move_reward(self,
                                 to_dest: bool = False,
                                 valid_link: bool = True,
                                 min_dist_before: int = 999,
                                 min_dist_after: int = 999) -> float:
        """
        计算树构建阶段移动奖励 (含距离引导)
        """
        p = self.params
        if not valid_link:
            return p.invalid_link * p.reward_scale

        reward = p.move_cost

        # 到达目的节点但未连接 (踩中)
        if to_dest:
            reward += p.dest_reached_bonus

        # 🔥 距离引导逻辑
        if min_dist_after < min_dist_before:
            # 靠近了: +5.0
            reward += p.guidance_closer_rate
        elif min_dist_after > min_dist_before:
            # 远离了: -2.0
            reward += p.guidance_farther_penalty
        else:
            # 平移/原地: -0.5
            reward += p.guidance_idle_penalty

        return reward * p.reward_scale

    def compute_frequency_penalty(self, visit_count: int, is_hub: bool = False) -> float:
        """计算频次惩罚"""
        p = self.params
        if is_hub or visit_count <= p.freq_penalty_threshold:
            return 0.0

        excess = visit_count - p.freq_penalty_threshold
        # 例如: 访问 6 次 (超 1 次) -> -2.0
        return (p.freq_penalty_rate * excess) * p.reward_scale

    # =========================================================================
    # 超时与结算
    # =========================================================================
    def compute_timeout_reward(self,
                               in_vnf_phase: bool,
                               connected_count: int = 0,
                               total_dests: int = 1) -> float:
        """计算超时奖励/惩罚"""
        p = self.params

        # 第一阶段就挂了
        if in_vnf_phase:
            return p.vnf_timeout_penalty * p.reward_scale

        # 第二阶段超时，根据进度给分
        ratio = connected_count / total_dests if total_dests > 0 else 0

        if ratio >= p.timeout_high_progress_threshold:
            # 高进度 (>80%)，虽败犹荣，给奖励
            return (p.timeout_bonus_rate * ratio) * p.reward_scale
        else:
            # 低进度，惩罚
            penalty = min(100.0, p.timeout_penalty_rate * (1.0 - ratio))
            return -penalty * p.reward_scale

    # =========================================================================
    # 统一调用接口 (可选)
    # =========================================================================
    def get_reward(self, phase: str, **kwargs) -> float:
        """
        统一接口，根据 phase 自动分发
        :param phase: 'vnf_deploy', 'vnf_move', 'tree_connect', 'tree_move', 'timeout'
        """
        if phase == 'vnf_deploy':
            return self.compute_vnf_deploy_reward(**kwargs)
        elif phase == 'vnf_move':
            return self.compute_vnf_move_reward(**kwargs)
        elif phase == 'tree_connect':
            return self.compute_tree_connection_reward(**kwargs)
        elif phase == 'tree_move':
            return self.compute_tree_move_reward(**kwargs)
        elif phase == 'timeout':
            return self.compute_timeout_reward(**kwargs)
        elif phase == 'penalty':
            # 通用惩罚 (如非法动作)
            t = kwargs.get('type', 'invalid_action')
            val = getattr(self.params, t, -10.0)
            return val * self.params.reward_scale
        return 0.0

    # =========================================================================
    # 辅助功能 (保存/加载/诊断)
    # =========================================================================
    def record_step(self, reward: float):
        """记录单步奖励到 buffer (可选)"""
        self._buffer_reward += reward

    def finish_request(self, final_reward: float = 0.0):
        """请求结束，归档总奖励"""
        total = self._buffer_reward + final_reward
        self.reward_history.append(total)
        self._buffer_reward = 0.0

        if len(self.reward_history) > self.history_len:
            self.reward_history.pop(0)

    def save(self, path: str) -> bool:
        """保存状态"""
        try:
            state = {
                "params": asdict(self.params),
                "reward_history": self.reward_history
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self, path: str) -> bool:
        """加载状态"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            if "params" in state:
                # 兼容性处理：只加载匹配的参数
                valid_keys = RewardCriticParams.__annotations__.keys()
                filtered = {k: v for k, v in state["params"].items() if k in valid_keys}
                self.params = RewardCriticParams(**filtered)
            self.reward_history = state.get("reward_history", [])
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def on_new_request(self):
        """兼容接口"""
        self._buffer_reward = 0.0

    def criticize(self, **kwargs):
        """兼容旧版接口 (仅返回 0，建议使用新接口)"""
        return 0.0