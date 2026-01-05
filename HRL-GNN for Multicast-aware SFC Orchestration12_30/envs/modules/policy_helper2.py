# envs/modules/policy_helper.py
import numpy as np
import random
import logging
import sys
import copy
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set

from .sfc_backup_system.backup_policy import BackupPolicy

try:
    # 尝试标准路径导入 (假设项目根目录在 sys.path 中)
    from core.expert.expert_msfce import MSFCE_Solver
except ImportError:
    try:
        # 如果在根目录 (兼容旧习惯)
        from expert_msfce import MSFCE_Solver
    except ImportError:
        # 尝试动态添加路径
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # envs/modules/policy_helper.py -> ... -> root
        sys.path.append(str(project_root))

        try:
            from core.expert.expert_msfce import MSFCE_Solver
        except ImportError:
            # 最后的占位符 (防止IDE报错，运行时会打印错误)
            class MSFCE_Solver:
                def __init__(self, *args, **kwargs):
                    self.node_num = 28;
                    self.link_num = 100;
                    self.type_num = 8;
                    self.k_path = 5

                @property
                def k_path_count(self): return 5

                def solve_request_for_expert(self, *args): return None, []

logger = logging.getLogger(__name__)
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
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)

        expert_db_path = input_dir / "US_Backbone_path.mat"
        if not expert_db_path.exists():
            logger.warning(f"⚠️ 专家数据库未找到: {expert_db_path}，尝试使用默认路径")
            # 尝试在 data/input_dir 找
            expert_db_path = input_dir.parent / "input_dir" / "US_Backbone_path.mat"

        # 初始化专家
        self.expert = MSFCE_Solver(expert_db_path, topo, dc_nodes, capacities)

        # 2. 🔥 [修复核心] 正确初始化备份策略
        # 必须传入 expert 实例以及从 expert 获取的网络维度参数
        self.backup_policy = BackupPolicy(
            expert=self.expert,  # 必须传 expert 实例
            n=self.expert.node_num,  # 节点数
            L=self.expert.link_num,  # 链路数
            K_vnf=self.expert.type_num,  # VNF类型数 (这就是报错缺少的 K_vnf)
            dc_nodes=dc_nodes
        )

        # 3. 常用参数缓存
        self.K_path = getattr(self.expert, 'k_path_count', 5)
        self.expert_randomness = 0.1  # 专家策略的随机扰动概率
        self._eval_cache = {}  # 专家评估结果缓存

        self._expert_cache = {}
    def clear_cache(self):
        """每步开始前清空缓存"""
        self._eval_cache.clear()

    # =========================================================================
    # 核心功能：获取部署方案 (Expert + Backup)
    # =========================================================================
    def get_best_plan(self, request, network_state, goal_dest_idx, k_idx, i_idx,
                      current_tree, nodes_on_tree, path_manager):
        """
        获取部署方案（带调试打印）
        """
        req_id = request['id']

        # --- 调试打印 START ---
        # 只打印前几个请求的调试信息，防止刷屏
        debug_mode = (req_id < 5)
        if debug_mode:
            print(f"\n[DEBUG] Req {req_id} | GoalIdx: {goal_dest_idx}")
        # --- 调试打印 END ---

        # 1. 尝试从专家缓存中提取路径
        if req_id in self._expert_cache and self._expert_cache[req_id] is not None:
            expert_info = self._expert_cache[req_id]

            # 计算目标节点 ID
            # goal_dest_idx 是 request['dest'] 列表的下标
            if 0 <= goal_dest_idx < len(request['dest']):
                dest_node_0 = request['dest'][goal_dest_idx]  # 0-based
                dest_node_1 = dest_node_0 + 1  # 1-based (专家可能用的Key)
            else:
                if debug_mode: print(f"  ❌ Goal Index {goal_dest_idx} out of range!")
                return False, None, False, "invalid_goal_idx"

            # 获取专家计算的路径映射
            tree_info = expert_info.get('tree')
            if not tree_info or 'paths_map' not in tree_info:
                if debug_mode: print("  ❌ Cache hit but 'paths_map' is missing!")
                return False, None, False, "expert_data_corrupt"

            paths_map = tree_info['paths_map']
            if debug_mode: print(
                f"  Info: paths_map keys: {list(paths_map.keys())} | Looking for: {dest_node_1} (1-based) or {dest_node_0} (0-based)")

            # 尝试查找路径 (兼容 1-based 和 0-based Key)
            path_found = None

            # 优先尝试 1-based (因为专家通常用物理ID)
            if dest_node_1 in paths_map:
                path_found = paths_map[dest_node_1]
            # 其次尝试 0-based
            elif dest_node_0 in paths_map:
                path_found = paths_map[dest_node_0]

            if path_found is not None:
                # ✅ 找到了专家路径！
                # 专家路径通常是 [src, n1, n2, dst]
                # 这里的节点 ID 可能是 1-based，需要检查并转回 0-based

                # 简单判断：如果路径里有 >= 28 的数字，或者 == dest_node_1，说明是 1-based
                is_1based_path = any(n > 27 for n in path_found) or (dest_node_1 in path_found)

                if is_1based_path:
                    path_0based = [n - 1 for n in path_found]
                else:
                    path_0based = list(path_found)

                if debug_mode: print(f"  ✅ Path found: {path_0based}")

                # 构造部署计划
                # 使用 BackupPolicy 的工具函数来计算具体的 VNF 放置 (HVT)
                hvt_map = self.backup_policy.place_vnfs(request, path_0based)

                if hvt_map is None:
                    if debug_mode: print("  ❌ Path found but VNF placement failed (Resource shortage?)")
                    return False, None, False, "expert_path_resource_shortage"

                plan = {
                    'nodes': path_0based,
                    'new_path_full': path_0based,
                    'hvt': hvt_map,
                    # 占位符，防止报错
                    'tree': np.zeros(self.expert.link_num if hasattr(self.expert, 'link_num') else 100)
                }

                return True, plan, False, "expert_success"
            else:
                if debug_mode: print(f"  ❌ Target node not found in expert paths_map!")

        else:
            if debug_mode: print(f"  ⚠️ Req {req_id} not in expert cache (or None)")

        # 2. 如果专家没缓存，回退到备份策略
        if debug_mode: print("  -> Fallback to BackupPolicy")
        return self.backup_policy.find_path_and_deploy(
            request, network_state, goal_dest_idx, nodes_on_tree
        )
    # =========================================================================
    # 辅助功能：高层策略 (High Level)
    # =========================================================================
    def get_expert_candidates(self, request, network_state, unadded_dests,
                              current_tree, nodes_on_tree, top_k=5):
        """
        获取专家建议的高层目标
        """
        # 1. 获取专家方案 (含索引转换)
        expert_plan = self._run_expert_if_needed(request, network_state)

        if expert_plan is None:
            return []

        trajectory = expert_plan['trajectory']
        # trajectory 里的元素是 (d_idx, action_tuple, cost)

        # 2. 找出专家规划中，下一个还没完成的目标
        target_dest_idx = -1

        for item in trajectory:
            # item[0] 是 d_idx (request['dest']列表里的下标)
            d_idx = item[0]

            # 如果这个目标在“未完成列表”里，那就是它了
            if d_idx in unadded_dest_indices_set(unadded_dests):
                target_dest_idx = d_idx
                break

        if target_dest_idx == -1:
            return []

        # 3. 返回格式: [(目标下标, 置信度)]
        candidates = [(target_dest_idx, 10.0)]

        # 把其他未选的也加上，给 0 分
        for d in unadded_dests:
            if d != target_dest_idx:
                candidates.append((d, 0.0))

        return candidates
    def get_expert_high_level_goal(self, request, network_state, unadded_dests,
                                   current_tree, nodes_on_tree):
        """获取专家推荐的单一高层目标"""
        cands = self.get_expert_candidates(
            request, network_state, unadded_dests, current_tree, nodes_on_tree, top_k=1
        )
        if cands:
            return int(cands[0][0])
        # Fallback
        if unadded_dests:
            return int(next(iter(unadded_dests)))
        return 0

    def get_expert_high_level_labels(self, request, network_state, unadded_dests,
                                     current_tree, nodes_on_tree, top_k=5):
        """获取专家的高层策略标签 (用于 Phase 2)"""
        cands = self.get_expert_candidates(
            request, network_state, unadded_dests, current_tree, nodes_on_tree, top_k=top_k
        )
        if not cands:
            return [], [], 0

        ids = [int(c[0]) for c in cands]
        scores = [float(c[1]) for c in cands]
        return ids, scores, ids[0]

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
            for i in range(num_paths):
                for k in range(self.expert.k_path_count):
                    action_id = i * self.expert.k_path_count + k
                    valid_actions.append(action_id)

        return valid_actions if valid_actions else [0]

    def _get_path_for_i_idx(self, path_manager, current_tree, request, i_idx: int) -> List[int]:
        """根据索引从 PathManager 获取具体路径列表"""
        if not current_tree or not current_tree.get('paths_map'):
            return [request['source']] if request else [0]

        path = path_manager.get_path(i_idx)
        if path is None:
            # 如果索引越界，做回退处理
            if len(path_manager) > 0:
                return path_manager.get_path(0)
            return [request['source']]
        return path

    def decode_low_level_action(self, action: int, max_paths: int = 10) -> Tuple[int, int]:
        """
        解码低层动作为 (i_idx, k_idx)
        公式: action = i_idx * K_path + k_idx
        """
        k_idx = int(action % self.K_path)
        i_idx = int(action // self.K_path)
        # 限制在有效范围
        i_idx = i_idx % max_paths
        return i_idx, k_idx

    def expert_low_level_action(self) -> int:
        """返回上一步专家推荐的低层动作"""
        return getattr(self, 'last_expert_action', -1)

    # =========================================================================
    # 辅助功能：掩码计算 (Masks)
    # =========================================================================
    def get_high_level_candidate_mask(self, candidates: List[Tuple[int, float]], num_goals: int) -> np.ndarray:
        """生成高层候选动作掩码"""
        mask = np.zeros(num_goals, dtype=np.float32)
        for dest_idx, _ in candidates:
            if 0 <= dest_idx < num_goals:
                mask[dest_idx] = 1.0
        return mask

    def get_low_level_action_mask(self, path_manager, current_tree,
                                  num_actions: int) -> np.ndarray:
        """生成低层动作掩码"""
        mask = np.zeros(num_actions, dtype=np.float32)
        valid_actions = self.get_valid_low_level_actions(path_manager, current_tree)
        for action in valid_actions:
            if 0 <= action < num_actions:
                mask[action] = 1.0
        return mask

    def _run_expert_if_needed(self, request, network_state):
        req_id = request['id']

        # 1. 如果缓存里已经有这一单的方案，直接返回
        if req_id in self._expert_cache:
            return self._expert_cache[req_id]

        # 2. 转换请求：从环境索引(0~27) -> 专家索引(1~28)
        expert_req = copy.deepcopy(request)
        expert_req['source'] = int(request['source']) + 1  # 🔥 +1
        expert_req['dest'] = [int(d) + 1 for d in request['dest']]  # 🔥 +1

        # 3. 调用专家 (solve_request_for_expert 会计算全量部署方案)
        try:
            tree_info, trajectory = self.expert.solve_request_for_expert(expert_req, network_state)
        except Exception as e:
            logger.error(f"Expert solver crashed: {e}")
            tree_info, trajectory = None, []

        if tree_info is None:
            # 专家也解不出来
            self._expert_cache[req_id] = None
            return None

        # 4. 结果处理
        # trajectory 是专家建议的服务顺序，结构是 [(d_idx, action, cost), ...]
        # 我们只需要 d_idx (这是相对于 dest 列表的索引，不需要转换)
        # 但如果是节点 ID 列表，则需要 -1。MSFCE 返回的是 d_idx 列表。

        self._expert_cache[req_id] = {
            'tree': tree_info,
            'trajectory': trajectory  # 这里的 trajectory 存的是 plan
        }
        return self._expert_cache[req_id]

def unadded_dest_indices_set(unadded_dests):
    """兼容 set 或 list 输入"""
    if isinstance(unadded_dests, (list, tuple)):
        return set(unadded_dests)
    return unadded_dests

    # =========================================================================
    # 一阶段专属方法
    # =========================================================================

    # =========================================================================
    # 二阶段三阶段专属方法
    # =========================================================================