# reward_critic.py
"""
Reward Critic Module
修复记录:
1. ✅ criticize 方法增加 success 参数，解决 TypeError
2. ✅ 增加 **kwargs 兼容未来可能的参数扩展
"""
import logging
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class RewardCritic:
    """
    Scheme B Reward Critic (无抢跑版 + 接口兼容修复)
    """

    def __init__(self,
                 training_phase: int = 2,
                 epoch: int = 0,
                 max_epochs: int = 1200,
                 params: Optional[Dict[str, Any]] = None):

        self.phase = int(training_phase)
        self.epoch = int(epoch)
        self.max_epochs = int(max_epochs)

        # 参数配置
        default_params = {
            "subtask_success": 1.0,
            "subtask_failure": -1.0,
            "cost_weight": 0.5,
            "progress_weight": 0.5,
            "backup_penalty": 0.2,
            "full_accept_bonus": 20.0,
            "request_fail_penalty": 10.0,
            "step_clip": (-2.0, 2.0),
        }
        self.params = default_params if params is None else {**default_params, **params}

        # Buffer
        self._buffer_reward = 0.0
        self._request_active = False

        # Stats
        self.reward_history = []
        self.debug = False
        self.backup_success_rate = defaultdict(lambda: 0.5)
        self.backup_usage_count = defaultdict(int)

    # ---------------------------------------------------------
    # 生命周期
    # ---------------------------------------------------------
    def on_new_request(self):
        self._buffer_reward = 0.0
        self._request_active = True

    def on_request_done(self, full_accept: bool) -> float:
        if not self._request_active:
            # 防止重复调用返回 0
            return 0.0

        if full_accept:
            final_reward = self.params["full_accept_bonus"] + self._buffer_reward
        else:
            final_reward = -self.params["request_fail_penalty"]

        self._buffer_reward = 0.0
        self._request_active = False

        self.reward_history.append(final_reward)
        return float(final_reward)

    # ---------------------------------------------------------
    # 内部算分
    # ---------------------------------------------------------
    def _step_subreward(self, sub_task_completed, cost, progress, backup_used):
        p = self.params
        r = 0.0
        r += p["subtask_success"] if sub_task_completed else p["subtask_failure"]
        r -= p["cost_weight"] * float(np.clip(cost, 0.0, 1.0))
        if abs(progress) > 0.2:
            r += p["progress_weight"] * float(np.clip(progress, -1.0, 1.0))
        if backup_used:
            r -= p["backup_penalty"]

        lo, hi = p["step_clip"]
        r = float(np.clip(r, lo, hi))

        if self._request_active:
            self._buffer_reward += r
        return r

    # ---------------------------------------------------------
    # 🔥🔥🔥 核心修复点：增加参数兼容 🔥🔥🔥
    # ---------------------------------------------------------
    def criticize(self,
                  sub_task_completed: bool = False,
                  cost: float = 0.0,
                  request_failed: bool = False,
                  progress_to_goal: float = 0.0,
                  backup_used: bool = False,
                  backup_level: str = "unknown",
                  qos_violations: Optional[Dict[str, float]] = None,
                  failure_reason: Optional[str] = None,
                  agent_action: int = -1,
                  expert_action: int = -1,
                  state_novelty: float = 0.5,
                  expert_confidence: float = 1.0,
                  # 🚨 新增兼容参数
                  success: bool = False,
                  **kwargs) -> float:

        # 1. 自动初始化 (以防万一)
        if not self._request_active and not request_failed:
            self.on_new_request()

        # 兼容逻辑：如果没有传 sub_task_completed 但传了 success，则复用 success
        # (这取决于 sfc_env 的具体逻辑，通常 success 和 sub_task_completed 含义相近)
        if success and not sub_task_completed:
            sub_task_completed = True

        # 2. 计算并缓存 (只做这一件事)
        self._step_subreward(
            sub_task_completed=sub_task_completed,
            cost=cost,
            progress=progress_to_goal,
            backup_used=backup_used
        )

        # 3. 统一返回 0，坐等 Env 显式调用 on_request_done
        return 0.0

    # ---------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------
    def set_training_phase(self, phase: int, epoch: int = 0, max_epochs: int = None):
        pass

    def set_debug(self, on: bool):
        self.debug = on

    def get_reward_diagnostics(self):
        return {}

    # 增加 save/load 空方法，防止 Trainer 调用报错
    def save(self, path):
        pass

    def load(self, path):
        pass