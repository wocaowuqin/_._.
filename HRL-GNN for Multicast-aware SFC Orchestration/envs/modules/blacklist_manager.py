# 文件路径: core/env/blacklist_manager.py
# 新建这个文件

"""
黑名单管理器
负责管理资源不足/访问超限的节点，实现渐进冷却策略
"""

import logging
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)


class BlacklistManager:
    """
    黑名单管理器（无永久拉黑）

    核心功能：
    1. 失败节点加入黑名单（渐进冷却）
    2. 定期清理过期黑名单
    3. 应急清理（可用动作太少时）
    4. 服务释放时清理相关节点
    """

    def __init__(
            self,
            cooldown_base: int = 5,
            cooldown_max: int = 20,
            cooldown_multiplier: int = 2,
            cleanup_interval: int = 3,
            min_valid_actions: int = 3
    ):
        """
        初始化黑名单管理器

        Args:
            cooldown_base: 基础冷却时间（步数）
            cooldown_max: 最大冷却时间（步数）
            cooldown_multiplier: 失败次数增长系数
            cleanup_interval: 清理间隔（步数）
            min_valid_actions: 最小可用动作数（触发应急清理）
        """
        self._blacklist = {}  # {node_id: BlacklistInfo}

        # 冷却配置
        self._cooldown_base = cooldown_base
        self._cooldown_max = cooldown_max
        self._cooldown_multiplier = cooldown_multiplier

        # 清理配置
        self._cleanup_interval = cleanup_interval
        self._min_valid_actions = min_valid_actions

        # 计数器
        self._step_counter = 0
        self._last_cleanup_step = 0

    def reset(self):
        """重置黑名单（每个SFC开始时）"""
        self._blacklist.clear()
        self._step_counter = 0
        self._last_cleanup_step = 0
        logger.debug("🔄 黑名单已重置")

    def increment_step(self):
        """增加步数计数"""
        self._step_counter += 1

    def _calculate_cooldown(self, attempts: int) -> int:
        """
        计算冷却时间

        失败1次: 5步
        失败2次: 10步
        失败3次: 20步（上限）
        """
        cooldown = min(
            self._cooldown_base * (self._cooldown_multiplier ** (attempts - 1)),
            self._cooldown_max
        )
        return int(cooldown)

    def add_node(self, node_id: int, reason: str):
        """
        添加节点到黑名单

        Args:
            node_id: 节点ID
            reason: 失败原因（如"资源不足"、"访问超限"）
        """
        current_step = self._step_counter

        if node_id not in self._blacklist:
            # 首次失败
            attempts = 1
            cooldown = self._calculate_cooldown(attempts)

            self._blacklist[node_id] = {
                'first_fail_step': current_step,
                'last_fail_step': current_step,
                'attempts': attempts,
                'reason': reason,
                'cooldown_until': current_step + cooldown,
                'cooldown_duration': cooldown
            }

            logger.warning(
                f"⚠️ 节点{node_id}加入黑名单 "
                f"(原因: {reason}, 冷却{cooldown}步)"
            )
        else:
            # 再次失败
            info = self._blacklist[node_id]
            info['attempts'] += 1
            info['last_fail_step'] = current_step

            cooldown = self._calculate_cooldown(info['attempts'])
            info['cooldown_until'] = current_step + cooldown
            info['cooldown_duration'] = cooldown

            logger.warning(
                f"⚠️ 节点{node_id}第{info['attempts']}次失败 "
                f"(延长冷却至{cooldown}步)"
            )

    def is_blacklisted(self, node_id: int) -> bool:
        """
        检查节点是否在黑名单中

        Args:
            node_id: 节点ID

        Returns:
            True: 在黑名单且冷却中
            False: 不在黑名单或冷却已过期
        """
        if node_id not in self._blacklist:
            return False

        info = self._blacklist[node_id]
        return self._step_counter < info['cooldown_until']

    def get_blacklisted_nodes(self) -> Set[int]:
        """
        获取当前黑名单中的所有节点

        Returns:
            节点ID集合
        """
        return set(self._blacklist.keys())

    def clean_expired(self, resource_checker=None):
        """
        清理过期的黑名单条目

        Args:
            resource_checker: 可选的资源检查函数 func(node_id) -> bool

        Returns:
            移除的节点列表
        """
        current_step = self._step_counter
        removed = []

        for node_id in list(self._blacklist.keys()):
            info = self._blacklist[node_id]

            # 策略1: 冷却到期 → 强制移除
            if current_step >= info['cooldown_until']:
                del self._blacklist[node_id]
                removed.append(node_id)
                logger.info(
                    f"✅ 节点{node_id}冷却到期，移出黑名单 "
                    f"(冷却{current_step - info['last_fail_step']}步)"
                )

            # 策略2: 主动检查资源恢复（至少冷却了基础时间）
            elif resource_checker and current_step - info['last_fail_step'] >= self._cooldown_base:
                if resource_checker(node_id):
                    del self._blacklist[node_id]
                    removed.append(node_id)
                    logger.info(f"✅ 节点{node_id}资源已恢复，提前移出黑名单")

        if removed:
            logger.info(f"🔄 定期清理移除{len(removed)}个节点: {removed}")

        return removed

    def emergency_clean(self, resource_checker=None):
        """
        应急清理（可用动作太少时）

        策略：
        1. 按失败次数排序（少的优先）
        2. 按剩余冷却时间排序（短的优先）
        3. 移除前50%

        Args:
            resource_checker: 可选的资源检查函数

        Returns:
            移除的节点列表
        """
        if not self._blacklist:
            return []

        logger.warning(
            f"⚠️ 可用动作过少，触发应急清理 "
            f"(黑名单: {len(self._blacklist)}个节点)"
        )

        # 排序候选节点
        candidates = []
        current_step = self._step_counter

        for node_id, info in self._blacklist.items():
            remaining = max(0, info['cooldown_until'] - current_step)
            candidates.append((
                node_id,
                info['attempts'],  # 失败次数
                remaining  # 剩余冷却
            ))

        # 排序：失败少的优先，冷却短的优先
        candidates.sort(key=lambda x: (x[1], x[2]))

        # 移除前50%
        num_to_remove = max(1, len(candidates) // 2)
        removed = []

        for i in range(num_to_remove):
            node_id = candidates[i][0]

            # 可选：检查资源
            if resource_checker:
                if resource_checker(node_id):
                    del self._blacklist[node_id]
                    removed.append(node_id)
                    logger.info(f"🔄 应急移除节点{node_id} (资源充足)")
                else:
                    # 资源仍不足，但也移除（给机会）
                    del self._blacklist[node_id]
                    removed.append(node_id)
                    logger.warning(f"⚠️ 强制移除节点{node_id} (资源可能不足)")
            else:
                del self._blacklist[node_id]
                removed.append(node_id)

        if removed:
            logger.info(f"🔄 应急清理移除{len(removed)}个节点: {removed}")

        return removed

    def on_service_release(self, released_nodes: Set[int], resource_checker=None):
        """
        服务释放时清理黑名单

        Args:
            released_nodes: 释放的节点集合
            resource_checker: 资源检查函数

        Returns:
            移除的节点列表
        """
        if not released_nodes:
            return []

        removed = []

        for node_id in released_nodes:
            if node_id in self._blacklist:
                # 服务释放，检查资源是否恢复
                if resource_checker is None or resource_checker(node_id):
                    info = self._blacklist[node_id]
                    del self._blacklist[node_id]
                    removed.append(node_id)

                    logger.info(
                        f"✅ 节点{node_id}服务释放，资源恢复，移出黑名单 "
                        f"(曾失败{info['attempts']}次)"
                    )

        if removed:
            logger.info(f"🔄 服务释放后清理{len(removed)}个节点")

        return removed

    def should_clean(self) -> bool:
        """判断是否应该执行定期清理"""
        return self._step_counter - self._last_cleanup_step >= self._cleanup_interval

    def mark_cleaned(self):
        """标记已执行清理"""
        self._last_cleanup_step = self._step_counter

    def get_info(self) -> Dict:
        """
        获取黑名单信息（用于调试）

        Returns:
            包含黑名单详情的字典
        """
        current_step = self._step_counter
        details = {}

        for node_id, info in self._blacklist.items():
            remaining = max(0, info['cooldown_until'] - current_step)
            details[node_id] = {
                'attempts': info['attempts'],
                'reason': info['reason'],
                'remaining_cooldown': remaining,
                'total_cooldown': info['cooldown_duration'],
                'age_steps': current_step - info['first_fail_step']
            }

        return {
            'total': len(self._blacklist),
            'nodes': list(self._blacklist.keys()),
            'details': details
        }