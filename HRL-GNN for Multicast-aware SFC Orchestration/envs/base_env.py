# modules/base_env.py
import gym
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict


class BaseEnv(gym.Env, ABC):
    """
    SFC 分层强化学习环境抽象基类

    所有具体环境（Flat、GNN、测试版等）必须继承此类并实现以下接口。
    注意：由于是分层 RL，标准 step() 仅作为兼容接口，实际交互推荐使用
          step_high_level() 和 step_low_level()。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config

    @abstractmethod
    def load_dataset(self, phase: str) -> bool:
        """
        加载指定阶段的数据集
        Returns:
            bool: 加载是否成功
        """
        pass

    @abstractmethod
    def reset(self, phase: str = "phase3") -> Any:
        """
        重置环境并开始新 episode
        Args:
            phase: 数据阶段（如 "phase3"）
        Returns:
            initial observation
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        标准 Gym step 接口（兼容性保留）
        在分层环境中通常映射到 low-level 动作
        """
        pass

    # ====================== 可选：建议添加分层接口（提高类型提示） ======================
    # 虽然 Python 不强制，但加上能让 IDE 和开发者更清楚预期行为

    def step_high_level(self, goal_idx: int) -> Tuple[Any, float, bool, Dict]:
        """
        高层动作：选择下一个要接入的目标分支
        """
        raise NotImplementedError("Subclasses should implement step_high_level if using HRL")

    def step_low_level(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        低层动作：选择路径与VNF部署方案
        Returns:
            obs, reward, sub_done（当前目标完成）, req_done（整个请求完成）, info
        """
        raise NotImplementedError("Subclasses should implement step_low_level if using HRL")

    def get_state(self) -> Any:
        """统一获取当前状态（Flat 或 Graph）"""
        raise NotImplementedError