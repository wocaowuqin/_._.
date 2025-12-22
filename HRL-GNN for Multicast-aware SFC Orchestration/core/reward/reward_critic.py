# reward_critic.py
"""
Reward Critic Module for Multicast VNF Mapping (完整修复版)

修复记录:
1. ✅ 添加全局奖励缩放 (reward_scale = 0.1)
2. ✅ 修复归一化基准 (max_cpu=100, max_bw=1000)
3. ✅ 调整奖励比例 (full_accept_bonus=5.0)
4. ✅ 增强惩罚力度 (invalid_penalty=-2.0)
5. ✅ 添加 hops 参数支持
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
    """奖励函数参数配置 (完整修复版)"""

    # ========================================
    # 核心权重
    # ========================================
    w_cpu: float = 0.33
    w_bw: float = 0.34
    w_hop: float = 0.33

    # ========================================
    # 🔥 单步奖励缩放
    # ========================================
    step_scale: float = 0.25

    # ========================================
    # 🔥🔥🔥 全局奖励缩放（关键修复）
    # ========================================
    reward_scale: float = 0.1

    # 作用：所有奖励最后乘以此值
    # 效果：Episode 奖励从 320 降到 32
    #       Loss 从 15 降到 5

    # ========================================
    # 惩罚参数
    # ========================================
    backup_penalty: float = 0.1
    invalid_penalty: float = -2.0  # 加大惩罚
    loop_penalty: float = -1.5  # 加大惩罚

    # ========================================
    # 任务结算
    # ========================================
    subtask_success: float = 2.0
    full_accept_bonus: float = 5.0  # 从 20.0 降到 5.0
    request_fail_penalty: float = 3.0

    # ========================================
    # 归一化基准（已修复）
    # ========================================
    max_cpu: float = 80.0  # 恢复正确值
    max_bw: float = 80.0  # 恢复正确值
    max_hops: float = 10.0

    # ========================================
    # 裁剪范围
    # ========================================
    step_clip: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class CriticDiagnostics:
    """诊断信息数据结构"""
    total_rewards: List[float]
    avg_reward: float
    request_count: int
    success_rate: float
    avg_bandwidth: float
    avg_hops: float


class RewardCritic:
    """
    Multicast VNF Reward Critic (完整修复版)
    """

    def __init__(self,
                 training_phase: int = 1,
                 params: Optional[Union[Dict[str, Any], RewardCriticParams]] = None):

        self.phase = int(training_phase)

        # 参数初始化
        if params is None:
            self.params = RewardCriticParams()
        elif isinstance(params, dict):
            self.params = RewardCriticParams(**params)
        else:
            self.params = params

        # 运行时状态
        self._buffer_reward = 0.0
        self._request_active = False
        self._current_request_id: Optional[str] = None

        # 统计数据容器
        self.reward_history: List[float] = []
        self.bw_history: List[float] = []
        self.hop_history: List[int] = []

        self.debug = False
        self._cached_params = asdict(self.params)

    # ---------------------------------------------------------
    # 核心计算逻辑
    # ---------------------------------------------------------
    def _normalize(self, value: float, max_val: float) -> float:
        """归一化工具"""
        if max_val <= 0:
            return 0.0
        return float(np.clip(value / max_val, 0.0, 1.0))

    def _calculate_step_reward(self,
                               cpu_remain: float = 0.0,
                               bandwidth: float = 0.0,
                               hops: int = 1,
                               is_node_action: bool = False,
                               backup_used: bool = False,
                               is_loop: bool = False) -> float:
        """
        计算单步奖励（对齐论文公式）

        论文公式：
        Rstep = β1·cpu + β2·bw - β3·hops
        """
        p = self.params

        # 1. 基础惩罚检查
        if is_loop:
            return p.loop_penalty
        if backup_used:
            return -p.backup_penalty

        # 2. 归一化
        norm_cpu = self._normalize(cpu_remain, p.max_cpu)
        norm_bw = self._normalize(bandwidth, p.max_bw)
        norm_hops = self._normalize(hops, p.max_hops)

        # 3. 计算原始奖励
        if is_node_action:
            # Meta Controller: 重点关注节点资源
            raw_reward = p.w_cpu * norm_cpu
        else:
            # Intrinsic Controller: 带宽 - 跳数成本
            raw_reward = (p.w_bw * norm_bw) - (p.w_hop * norm_hops)

        # 4. 单步奖励缩放
        reward = raw_reward * p.step_scale

        # 5. 裁剪
        lo, hi = p.step_clip
        return float(np.clip(reward, lo, hi))

    # ---------------------------------------------------------
    # 对外接口
    # ---------------------------------------------------------
    def criticize(self,
                  request_failed: bool = False,
                  sub_task_completed: bool = False,
                  request_completed: bool = False,

                  # 关键状态参数
                  cpu_remain: float = 0.0,
                  bandwidth: float = 0.0,
                  hops: int = 1,
                  is_meta_step: bool = False,
                  is_loop: bool = False,
                  backup_used: bool = False,

                  **kwargs) -> float:
        """
        计算并返回当前步的奖励（包含全局归一化）
        """
        # 1. 自动初始化请求
        if not self._request_active:
            self.on_new_request()

        # 2. 处理请求失败
        if request_failed:
            fail_reward = -self.params.request_fail_penalty
            # 🔥 应用全局缩放
            fail_reward *= self.params.reward_scale

            if self.debug:
                logger.debug(f"请求失败惩罚: {fail_reward}")

            self.on_request_done(success=False, final_reward_override=fail_reward)
            return fail_reward

        # 3. 计算单步奖励
        step_reward = self._calculate_step_reward(
            cpu_remain=cpu_remain,
            bandwidth=bandwidth,
            hops=hops,
            is_node_action=is_meta_step,
            backup_used=backup_used,
            is_loop=is_loop
        )

        # 4. 处理子任务完成
        if sub_task_completed:
            step_reward += self.params.subtask_success

        # 5. 处理整个请求完成
        if request_completed:
            step_reward += self.params.full_accept_bonus

            # 记录统计信息
            if 'path_min_bw' in kwargs:
                self.bw_history.append(kwargs['path_min_bw'])
            if 'path_hops' in kwargs:
                self.hop_history.append(kwargs['path_hops'])

        # ========================================
        # 🔥🔥🔥 关键：全局奖励归一化
        # ========================================
        step_reward *= self.params.reward_scale

        # 6. 累加 Buffer（归一化后的值）
        self._buffer_reward += step_reward

        # 7. 如果请求完成，归档
        if request_completed:
            self.on_request_done(success=True)

        if self.debug and abs(step_reward) > 0.001:
            logger.debug(f"Step: {step_reward:.4f} (CPU:{cpu_remain:.1f}, BW:{bandwidth:.1f}, Hops:{hops})")

        return step_reward

    # ---------------------------------------------------------
    # 生命周期管理
    # ---------------------------------------------------------
    def on_new_request(self, request_id: Optional[str] = None) -> None:
        """新回合开始"""
        self._buffer_reward = 0.0
        self._request_active = True
        self._current_request_id = request_id or f"req_{len(self.reward_history)}"

    def on_request_done(self, success: bool, final_reward_override: Optional[float] = None) -> float:
        """回合结束，归档数据"""
        final_reward = final_reward_override if final_reward_override is not None else self._buffer_reward
        self.reward_history.append(final_reward)
        self._request_active = False
        self._current_request_id = None
        return final_reward

    # ---------------------------------------------------------
    # 辅助与诊断
    # ---------------------------------------------------------
    def get_reward_diagnostics(self) -> CriticDiagnostics:
        """获取诊断数据"""
        count = len(self.reward_history)
        if count == 0:
            return CriticDiagnostics([], 0, 0, 0, 0, 0)

        positive_rewards = [r for r in self.reward_history if r > 0]

        return CriticDiagnostics(
            total_rewards=self.reward_history.copy(),
            avg_reward=float(np.mean(self.reward_history)),
            request_count=count,
            success_rate=len(positive_rewards) / count,
            avg_bandwidth=float(np.mean(self.bw_history)) if self.bw_history else 0.0,
            avg_hops=float(np.mean(self.hop_history)) if self.hop_history else 0.0
        )

    def save(self, path: str) -> bool:
        """保存训练状态"""
        try:
            state = {
                "params": asdict(self.params),
                "reward_history": self.reward_history,
                "bw_history": self.bw_history,
                "hop_history": self.hop_history
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self, path: str) -> bool:
        """加载训练状态"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            if "params" in state:
                self.params = RewardCriticParams(**state["params"])
            self.reward_history = state.get("reward_history", [])
            self.bw_history = state.get("bw_history", [])
            self.hop_history = state.get("hop_history", [])
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False