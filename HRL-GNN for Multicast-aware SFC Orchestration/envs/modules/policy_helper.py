# envs/modules/policy_helper.py
import numpy as np
import logging
import sys
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .sfc_backup_system.backup_policy import BackupPolicy

# ==========================================================
# Expert Import
# ==========================================================
try:
    from core.expert.expert_msfce import MSFCE_Solver
except ImportError:
    try:
        from expert_msfce import MSFCE_Solver
    except ImportError:
        # 动态添加路径适配
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        sys.path.append(str(project_root))
        try:
            from core.expert.expert_msfce import MSFCE_Solver
        except ImportError:
            # 占位符防止 IDE 报错
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


class PolicyHelper:
    """
    策略辅助模块：
    - Phase1：只允许专家规划（禁止 backup / 执行）
    - Phase2：专家引导 + backup
    - Phase3：纯 RL + backup
    """

    def __init__(self, input_dir, topo, dc_nodes, capacities):
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)

        expert_db_path = input_dir / "US_Backbone_path.mat"
        if not expert_db_path.exists():
            expert_db_path = input_dir.parent / "input_dir" / "US_Backbone_path.mat"

        # Expert
        self.expert = MSFCE_Solver(expert_db_path, topo, dc_nodes, capacities)

        # Backup Policy
        self.backup_policy = BackupPolicy(
            expert=self.expert,
            n=getattr(self.expert, 'node_num', 28),
            L=getattr(self.expert, 'link_num', 45),
            K_vnf=getattr(self.expert, 'type_num', 8),
            dc_nodes=dc_nodes,
        )

        self.K_path = getattr(self.expert, "k_path_count", 5)
        self._expert_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def clear_cache(self):
        self._expert_cache.clear()

    # ==========================================================
    # Phase 1 ONLY —— 专家全局规划
    # ==========================================================
    def get_expert_plan(self, request, network_state):
        """
        Phase1 唯一允许的接口：获取专家完整规划
        """
        expert_info = self._run_expert_if_needed(request, network_state)
        if expert_info is None:
            return None
        return {
            "tree": expert_info["tree"],
            "trajectory": expert_info["trajectory"],
        }

    def _run_expert_if_needed(self, request, network_state):
        req_id = request['id']
        if req_id in self._expert_cache:
            return self._expert_cache[req_id]

        N = self.expert.node_num  # = 28

        # ===== source =====
        src0 = int(request['source'])
        src1 = src0 + 1
        if not (1 <= src1 <= N):
            logger.error(f"[Expert] Invalid source after convert: {src1}")
            self._expert_cache[req_id] = None
            return None

        # ===== dests =====
        dests1 = []
        for d0 in request['dest']:
            d0 = int(d0)

            # 🔥 关键：先判断它是不是已经 1-based
            if 1 <= d0 <= N:
                d1 = d0
            else:
                d1 = d0 + 1

            if not (1 <= d1 <= N):
                logger.warning(
                    f"[Expert] Drop invalid dest: raw={d0}, conv={d1}, valid=[1,{N}]"
                )
                continue

            dests1.append(d1)

        if len(dests1) == 0:
            logger.warning(f"[Expert] No valid dests for req {req_id}")
            self._expert_cache[req_id] = None
            return None

        expert_req = {
            **request,
            "source": src1,
            "dest": dests1
        }

        try:
            tree_info, trajectory = self.expert.solve_request_for_expert(
                expert_req, network_state
            )
        except Exception as e:
            logger.error(f"[Expert] Solver crashed: {e}")
            self._expert_cache[req_id] = None
            return None

        if tree_info is None:
            self._expert_cache[req_id] = None
            return None

        self._expert_cache[req_id] = {
            "tree": tree_info,
            "trajectory": trajectory
        }
        return self._expert_cache[req_id]

    # ==========================================================
    # Phase 2 / Phase 3 —— 执行接口
    # ==========================================================
    def get_best_plan(
            self,
            request,
            network_state,
            goal_dest_idx,
            k_idx,
            i_idx,
            current_tree,
            nodes_on_tree,
            path_manager,
            unadded_dest_indices=None,
    ):
        """
        Phase2/3 使用：
        先尝试专家缓存路径，失败再使用 backup 策略兜底
        """
        req_id = request["id"]

        # 1. 尝试专家缓存
        if req_id in self._expert_cache and self._expert_cache[req_id] is not None:
            expert_info = self._expert_cache[req_id]
            paths_map = expert_info["tree"].get("paths_map", {})

            # 确定正确的目标索引
            if unadded_dest_indices is not None:
                try:
                    # goal_dest_idx 是 unadded_dest_indices 集合的相对下标
                    # 我们需要取回 request['dest'] 的绝对下标
                    real_idx = list(unadded_dest_indices)[goal_dest_idx]
                except Exception:
                    return False, None, False, "index_error"
            else:
                real_idx = goal_dest_idx

            # 获取目标节点 ID (Env 0-based)
            target_node_0 = request["dest"][real_idx]
            target_node_1 = target_node_0 + 1  # Expert Key (1-based)

            # 查找路径
            path = paths_map.get(target_node_1) or paths_map.get(target_node_0)

            if path is not None:
                # 转换路径节点回 0-based
                # 启发式判断：如果路径包含 >27 的节点或包含 1-based 目标，则判定为 1-based
                is_1based = any(n > 27 for n in path) or (target_node_1 in path)

                if is_1based:
                    path_0based = [n - 1 for n in path]
                else:
                    path_0based = list(path)

                # 计算 VNF 放置
                hvt_map = self.backup_policy.place_vnfs(request, path_0based)
                if hvt_map is not None:
                    plan = {
                        "nodes": path_0based,
                        "new_path_full": path_0based,
                        "hvt": hvt_map,
                        "tree": np.zeros(getattr(self.expert, 'link_num', 100)),
                    }
                    return True, plan, False, "expert_success"

        # 2. 回退到 Backup Policy
        return self.backup_policy.find_path_and_deploy(
            request, network_state, goal_dest_idx, nodes_on_tree
        )

    # ==========================================================
    # 兼容性接口 (Env 依赖)
    # ==========================================================
    def get_expert_candidates(self, request, network_state, unadded_dests,
                              current_tree, nodes_on_tree, top_k=5):
        """env.get_expert_high_level_candidates 依赖此接口"""
        expert_info = self._run_expert_if_needed(request, network_state)
        if expert_info is None:
            return []

        trajectory = expert_info['trajectory']
        target_idx = -1

        # 兼容 set 或 list
        unadded_set = set(unadded_dests) if isinstance(unadded_dests, (list, tuple)) else unadded_dests

        for item in trajectory:
            d_idx = item[0]
            if d_idx in unadded_set:
                target_idx = d_idx
                break

        if target_idx == -1:
            return []

        candidates = [(target_idx, 10.0)]
        for d in unadded_dests:
            if d != target_idx:
                candidates.append((d, 0.0))
        return candidates

    def expert_low_level_action(self):
        """env.expert_low_level_action 依赖此接口 (Phase1 返回无效动作)"""
        return -1

    # ==========================================================
    # Masks / Decoding
    # ==========================================================
    def decode_low_level_action(self, action, max_paths=10):
        return action // self.K_path, action % self.K_path

    def get_high_level_candidate_mask(self, candidates, num_goals):
        mask = np.zeros(num_goals, dtype=np.float32)
        for idx, _ in candidates:
            if 0 <= idx < num_goals:
                mask[idx] = 1.0
        return mask

    def get_low_level_action_mask(self, *args, **kwargs):
        return np.ones(100, dtype=np.float32)