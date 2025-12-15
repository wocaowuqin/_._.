import numpy as np
import random
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from .sfc_backup_system.backup_policy import BackupPolicy
# 尝试导入专家模块，如果不存在则定义占位符
try:
    from expert_msfce import MSFCE_Solver
except ImportError:
    # 占位符，防止IDE报错或在无Expert环境下运行
    class MSFCE_Solver:
        def __init__(self, *args, **kwargs): pass

        @property
        def k_path_count(self): return 5

        def _calc_eval(self, *args): return -np.inf, [], [], [], False, 0, 0, {}

        def _calc_atnp(self, *args): return {}, -np.inf, [], {}

logger = logging.getLogger(__name__)


class PolicyHelper:
    """
    策略辅助模块：
    1. 封装专家系统 (Expert System) 的复杂计算
    2. 封装备份策略 (Backup Policy) 的调用
    3. 提供高层/低层动作的掩码 (Mask) 计算
    """

    def __init__(self, input_dir, topo, dc_nodes, capacities):
        # 1. 初始化专家系统
        # 注意：Expert 需要读取 .mat 文件，路径需拼接
        expert_db_path = input_dir / "US_Backbone_path.mat"
        self.expert = MSFCE_Solver(expert_db_path, topo, dc_nodes, capacities)

        # 2. 初始化备份策略
        self.backup_policy = BackupPolicy(
            self.expert,
            self.expert.node_num,
            self.expert.link_num,
            self.expert.type_num,
            dc_nodes
        )

        # 3. 常用参数缓存
        self.K_path = self.expert.k_path_count
        self.expert_randomness = 0.1  # 专家策略的随机扰动概率
        self._eval_cache = {}  # 专家评估结果缓存

    def clear_cache(self):
        """每步开始前清空缓存"""
        self._eval_cache.clear()

    # =========================================================================
    # 核心功能：获取部署方案 (Expert + Backup)
    # =========================================================================
    def get_best_plan(self, request: Dict, network_state: Dict,
                      goal_dest_idx: int, k_idx: int, i_idx: int,
                      current_tree: Dict, nodes_on_tree: Set[int],
                      path_manager) -> Tuple[bool, Optional[Dict], bool, Optional[str]]:
        """
        尝试获取最佳部署方案：先问专家，不行再问备份。

        Returns:
            feasible (bool): 是否成功找到方案
            plan (dict): 具体的部署方案 (tree, hvt, new_path_full, placement)
            backup_used (bool): 是否使用了备份策略
            failure_reason (str): 失败原因
        """
        feasible = False
        plan = None
        backup_used = False
        failure_reason = None

        k = k_idx + 1  # Expert 使用 1-based index

        # --- 1. 尝试专家策略 (Expert) ---
        try:
            # Case A: 树为空 (Source -> Dest)
            if not current_tree['paths_map']:
                eval_val, paths, tree, hvt, feasible, _, _, placement_expert = \
                    self.expert._calc_eval(request, goal_dest_idx, k, network_state)

                if feasible:
                    plan = {
                        'tree': tree,
                        'hvt': hvt,
                        'new_path_full': paths,
                        'placement': placement_expert
                    }

            # Case B: 树已存在 (Tree -> Dest)
            else:
                # 从 PathManager 获取路径
                conn_path = self._get_path_for_i_idx(path_manager, current_tree, request, i_idx)

                plan, _, _, _ = self.expert._calc_atnp(
                    current_tree, conn_path, goal_dest_idx, network_state, nodes_on_tree
                )
                feasible = plan.get('feasible', False) if plan else False

        except Exception as e:
            feasible = False
            failure_reason = "expert_error"
            # logger.debug(f"Expert calculation failed: {e}")

        # --- 2. 尝试备份策略 (Backup) ---
        if not feasible:
            backup_used = True

            # 构建备份所需的状态副本 (防止修改原数据)
            bk_state = network_state.copy()
            # BackupPolicy 需要 float 类型的 cpu/mem 字典
            bk_state['cpu'] = {i: float(c) for i, c in enumerate(network_state['cpu'])}
            bk_state['mem'] = {i: float(m) for i, m in enumerate(network_state['mem'])}

            self.backup_policy.set_current_request(request)
            self.backup_policy.set_current_tree(list(nodes_on_tree))

            plan = self.backup_policy.get_backup_plan(goal_dest_idx, bk_state)

            if plan and plan.get('feasible'):
                feasible = True
            else:
                failure_reason = "resource_exhausted"

        return feasible, plan, backup_used, failure_reason

    # =========================================================================
    # 辅助功能：高层策略 (High Level)
    # =========================================================================
    def get_expert_candidates(self, request, network_state, unadded_dests,
                              current_tree, nodes_on_tree, top_k=5):
        """
        计算所有未完成目标节点的专家评分，返回候选列表
        """
        if not request or not unadded_dests:
            return []

        req_id = request['id']
        candidates = []

        # Stage 1: S -> d (树为空)
        if not current_tree['paths_map']:
            for d_idx in unadded_dests:
                best_eval = -float('inf')
                for k in range(1, self.K_path + 1):
                    # 使用缓存避免重复计算
                    cache_key = (req_id, d_idx, k)
                    if cache_key in self._eval_cache:
                        eval_val = self._eval_cache[cache_key]
                    else:
                        try:
                            eval_val, _, _, _, feasible, _, _, _ = self.expert._calc_eval(
                                request, d_idx, k, network_state
                            )
                            if not feasible or eval_val is None:
                                eval_val = -float('inf')
                        except:
                            eval_val = -float('inf')
                        self._eval_cache[cache_key] = eval_val

                    if eval_val > best_eval:
                        best_eval = eval_val

                if best_eval > -float('inf'):
                    candidates.append((d_idx, float(best_eval)))

        # Stage 2: Tree -> d (树已存在)
        else:
            for d_idx in unadded_dests:
                best_eval = -float('inf')
                # 遍历树上所有可能的连接路径
                for conn_path in current_tree['paths_map'].values():
                    try:
                        _, eval_val, _, _ = self.expert._calc_atnp(
                            current_tree, conn_path, d_idx, network_state, nodes_on_tree
                        )
                        if eval_val is not None:
                            best_eval = max(best_eval, eval_val)
                    except:
                        pass

            if best_eval > -float('inf'):
                candidates.append((d_idx, float(best_eval)))

        # 排序
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 增加随机性 (DAgger 常用技巧)
        if len(candidates) >= 2 and random.random() < self.expert_randomness:
            candidates[0], candidates[1] = candidates[1], candidates[0]

        return candidates[:top_k]

    def get_expert_high_level_goal(self, request, network_state, unadded_dests,
                                   current_tree, nodes_on_tree):
        """获取专家推荐的单一高层目标"""
        cands = self.get_expert_candidates(
            request, network_state, unadded_dests, current_tree, nodes_on_tree, top_k=1
        )
        if cands:
            return int(cands[0][0])
        # Fallback: 如果专家算不出来，就选第一个未完成的
        if unadded_dests:
            return int(next(iter(unadded_dests)))
        return 0

    # =========================================================================
    # 辅助功能：低层策略 (Low Level)
    # =========================================================================
    def get_valid_low_level_actions(self, path_manager, current_tree):
        """获取当前状态下所有合法的低层动作索引"""
        valid_actions = []

        # 如果树为空，所有 K_path (0~4) 都是合法的 (从源点出发)
        if not current_tree or not current_tree.get('paths_map'):
            for k in range(self.expert.k_path_count):
                valid_actions.append(k)
        else:
            # 否则，动作 = PathIdx * K_path + k
            num_paths = max(1, len(path_manager))
            # 限制 PathManager 中的路径数量，防止动作空间爆炸
            # (假设最大动作数 NB_LOW_LEVEL_ACTIONS 足够覆盖)
            for i in range(num_paths):
                for k in range(self.expert.k_path_count):
                    action_id = i * self.expert.k_path_count + k
                    valid_actions.append(action_id)

        return valid_actions if valid_actions else [0]

    def _get_path_for_i_idx(self, path_manager, current_tree, request, i_idx: int) -> List[int]:
        """根据索引从 PathManager 获取具体路径列表"""
        if not current_tree or not current_tree['paths_map']:
            return [request['source']] if request else [0]

        path = path_manager.get_path(i_idx)
        if path is None:
            # 如果索引越界，做回退处理 (例如取第0条路径)
            if len(path_manager) > 0:
                return path_manager.get_path(0)
            return [request['source']]
        return path

    # =========================================================================
    # 辅助功能：掩码计算 (Masks)
    # =========================================================================
    def get_action_masks(self, num_high: int, num_low: int,
                         high_cands: List[Tuple[int, float]],
                         low_valid: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算高层和低层的 Action Mask
        Returns:
            high_mask: [NB_HIGH_LEVEL_GOALS]
            low_mask: [NB_LOW_LEVEL_ACTIONS]
        """
        # 1. High-level mask
        high_mask = np.zeros(num_high, dtype=np.float32)
        for d_idx, _ in high_cands:
            if 0 <= d_idx < num_high:
                high_mask[d_idx] = 1.0

        # 2. Low-level mask
        low_mask = np.zeros(num_low, dtype=np.float32)
        for a in low_valid:
            if 0 <= a < num_low:
                low_mask[a] = 1.0

        return high_mask, low_mask

    # envs/modules/policy_helper.py

    def get_expert_high_level_labels(self, request, network_state, unadded_dests,
                                     current_tree, nodes_on_tree, top_k=5):
        """
        [遗漏补充] 获取专家的高层策略标签 (用于 Phase 2 模仿学习训练)
        返回: (ids, scores, best_id)
        """
        cands = self.get_expert_candidates(
            request, network_state, unadded_dests, current_tree, nodes_on_tree, top_k=top_k
        )
        if not cands:
            return [], [], 0

        ids = [int(c[0]) for c in cands]
        scores = [float(c[1]) for c in cands]
        return ids, scores, ids[0]

    def expert_low_level_action(self):
        """
        [遗漏补充] 返回上一步专家推荐的低层动作 (用于 DAgger 奖励)
        注意：这需要在 get_best_plan 中记录 self.last_expert_action
        """
        # 简单实现：如果没有记录，返回 -1
        return getattr(self, 'last_expert_action', -1)