# modules/core_env.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC_HIRL_Env - 完整可运行的主环境类（分层强化学习 + 多播感知）
已完全模块化，职责清晰，兼容 Flat 和 GNN 两种状态表示
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import random
import gym
from gym import spaces

# 导入自定义模块
from envs.modules.resource import ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.path_manager import PathManager
from envs.modules.event_handler import EventHandler
from envs.modules.policy_helper import PolicyHelper
from envs.modules.failure_visualizer import FailureVisualizer
from core.reward.reward_critic import RewardCritic

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SFC_HIRL_Env(gym.Env):
    """
    分层强化学习 SFC 部署环境（主类）

    支持：
    - 高层目标选择（High-Level Goal Selection）
    - 低层路径+VNF部署（Low-Level Path & Placement）
    - 专家系统 + 备份策略兜底
    - VNF 共享、多播树动态扩展
    - 失败可视化、奖励批评家、详细指标统计
    - 可切换 Flat / GNN 状态表示
    """

    # =========================================================================
    # 1. 初始化与配置
    # =========================================================================
    def __init__(self, config: Dict, use_gnn: bool = False):
        super().__init__()

        self.cfg = config
        self.use_gnn = use_gnn

        # 基础路径与配置
        self.input_dir = Path(config['path']['input_dir'])
        self.failure_output_dir = Path(
            config['path'].get('failure_output_dir',
                               self.input_dir.parent / "out_failure")
        )
        os.makedirs(self.failure_output_dir, exist_ok=True)

        self.enable_render = config.get('render', {}).get(
            'enable_failure_visualization', False
        )

        # 核心模块初始化
        topo = config['topology']['matrix']
        dc_nodes = config['topology']['dc_nodes']
        capacities = config['capacities']

        self.resource_mgr = ResourceManager(topo, capacities, dc_nodes)
        self.data_loader = DataLoader(config)
        self.path_manager = PathManager(
            max_paths=config.get('env', {}).get('max_cached_paths', 10)
        )
        self.event_handler = EventHandler(self.resource_mgr)

        # 策略辅助（专家 + 备份）
        self.policy_helper = PolicyHelper(self.input_dir, topo, dc_nodes, capacities)

        # 可视化
        if self.enable_render:
            self.visualizer = FailureVisualizer(topo=topo)

        # 奖励批评家
        self.reward_critic = RewardCritic()

        # 环境参数
        self.num_nodes = self.resource_mgr.n
        self.K_vnf = self.resource_mgr.K_vnf
        self.NB_HIGH_LEVEL_GOALS = config.get('env', {}).get('nb_high_level_goals', 10)
        self.NB_LOW_LEVEL_ACTIONS = config.get('env', {}).get('nb_low_level_actions', 50)

        # 动作与观察空间
        self.high_level_action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.low_level_action_space = spaces.Discrete(self.NB_LOW_LEVEL_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        # 运行时状态
        self.current_request: Optional[Dict] = None
        self.current_tree: Dict[str, Any] = {
            'tree': np.zeros(self.resource_mgr.L),
            'hvt': np.zeros((self.num_nodes, self.K_vnf)),
            'paths_map': {}
        }
        self.unadded_dest_indices: Set[int] = set()  # 统一使用这个名字
        self.nodes_on_tree: Set[int] = set()
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.phase_done = False

        # 状态访问计数（用于探索奖励）
        self._state_visit_counter = {}

        # 进度跟踪（用于 RewardCritic）
        self._prev_dist = None

    # =========================================================================
    # 2. 数据加载与重置
    # =========================================================================
    def load_dataset(self, phase_or_req_file: str,
                     events_file: Optional[str] = None) -> bool:
        """
        加载数据集（兼容两种调用方式）

        方式1: load_dataset("phase3")
        方式2: load_dataset("phase3_requests.pkl", "phase3_events.pkl")
        """
        # 方式2：文件名方式（旧环境兼容）
        if events_file is not None:
            import pickle

            # 尝试多个可能的目录
            possible_dirs = [
                Path("generate_requests_depend_on_poisson/data_output"),
                self.cfg['path'].get('expert_data_dir', 'data/expert'),
                Path('data/expert'),
                Path('.')
            ]

            req_path = None
            evt_path = None

            for data_dir in possible_dirs:
                data_dir = Path(data_dir)
                test_req = data_dir / phase_or_req_file
                test_evt = data_dir / events_file

                if test_req.exists() and test_evt.exists():
                    req_path = test_req
                    evt_path = test_evt
                    logger.info(f"Found data files in: {data_dir}")
                    break

            if req_path is None:
                logger.error(f"Data files not found: {phase_or_req_file}, {events_file}")
                return False

            try:
                # 加载请求
                with open(req_path, 'rb') as f:
                    requests = pickle.load(f)

                # 加载事件
                with open(evt_path, 'rb') as f:
                    raw_events = pickle.load(f)

                # 格式化事件
                events = []
                for evt in raw_events:
                    events.append({
                        'arrive': np.array(evt.get('arrive', []), dtype=int).flatten(),
                        'leave': np.array(evt.get('leave', []), dtype=int).flatten()
                    })

                # 更新 DataLoader
                self.data_loader.requests = requests
                self.data_loader.req_map = {r['id']: r for r in requests}
                self.data_loader.events = events
                self.data_loader.total_steps = len(events)
                self.data_loader.reset()

                logger.info(f"✓ Loaded {len(requests)} requests, {len(events)} events")
                return True

            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                import traceback
                traceback.print_exc()
                return False

        # 方式1：阶段名方式（新环境）
        else:
            return self.data_loader.load_dataset(phase_or_req_file)

    def _reset_core(self):
        """
        环境级 reset（不加载数据集、不推进请求）
        """
        self.resource_mgr.reset()
        self.data_loader.reset()
        self.path_manager.reset()
        self.event_handler.reset()
        self.policy_helper.clear_cache()

        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self._state_visit_counter.clear()
        self._prev_dist = None
        self.phase_done = False

    def reset(self, *, seed=None, options=None):
        """
        Gym / Phase3 专用 reset
        """
        super().reset(seed=seed)

        phase = "phase3"
        if options is not None:
            phase = options.get("phase", phase)

        logger.info(f"[Env] reset() called, phase={phase}")

        # 核心清理
        self._reset_core()

        # 加载数据集（只在 reset 做）
        if not self.load_dataset(phase):
            raise RuntimeError(f"Failed to load dataset for {phase}")

        # 初始化第一个请求
        req, _ = self.reset_request()

        obs = self.get_state()
        info = {"phase": phase}

        return obs, info

    def reset_request(self) -> Tuple[Optional[Dict], Any]:
        """
        重置并推进到下一个请求
        Phase1 / Phase2 / Phase3 共用
        """

        # 清理 request 级缓存
        self.policy_helper.clear_cache()
        self.current_request = None
        self._prev_dist = None

        # 通知奖励模块
        self.reward_critic.on_new_request()

        # === 核心：推进时间直到出现 arrival ===
        while True:
            # 处理离开事件
            leaves = self.data_loader.get_current_leaves()
            self.event_handler.process_leaves(leaves)

            # 获取到达请求
            arrivals = self.data_loader.get_current_arrivals()
            if arrivals:
                self.current_request = arrivals[0]
                break

            # 推进时间
            self.data_loader.advance_time()

            # 数据集结束
            if self.data_loader.is_done():
                logger.info("[Env] No more requests in dataset")
                return None, self.get_state()

        # ===== 初始化请求级状态 =====
        self.total_requests_seen += 1
        req = self.current_request

        dests = req.get("dest", [])
        self.unadded_dest_indices = set(range(len(dests)))

        self.nodes_on_tree = {req["source"]}

        self.current_tree = {
            "id": req["id"],
            "tree": np.zeros(self.resource_mgr.L, dtype=np.float32),
            "hvt": np.zeros((self.resource_mgr.n, self.resource_mgr.K_vnf), dtype=np.float32),
            "paths_map": {}
        }

        self.path_manager.reset()

        return req, self.get_state()

    def _reset_current_request(self):
        """内部使用：获取下一个到达的请求并初始化树结构"""
        arrivals = self.data_loader.get_current_arrivals()
        self.current_request = arrivals[0] if arrivals else None

        if self.current_request is None:
            self.phase_done = True
            return

        self.total_requests_seen += 1
        req = self.current_request

        # 初始化未完成目标集合
        dests = req.get('dest', [])
        self.unadded_dest_indices = set(range(len(dests)))

        # 初始化树上的节点（1-based 转 0-based）
        self.nodes_on_tree = {req['source']}

        # 重置当前树
        self.current_tree = {
            'tree': np.zeros(self.resource_mgr.L, dtype=np.float32),
            'hvt': np.zeros((self.num_nodes, self.K_vnf), dtype=np.float32),
            'paths_map': {}
        }

        self.path_manager.reset()
        self.policy_helper.clear_cache()
        self._prev_dist = None

    # =========================================================================
    # 3. 状态获取接口
    # =========================================================================
    def get_state(self):
        """
        统一状态获取接口

        根据 use_gnn 返回 Flat 或 Graph 状态
        """
        if self.use_gnn:
            # GNN 模式：返回图状态
            return self.resource_mgr.get_graph_state(
                current_request=self.current_request,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree,
                served_dest_count=len(self.current_tree.get('paths_map', {})),
                sharing_strategy=0,  # 可以根据需要设置
                nb_high_goals=self.NB_HIGH_LEVEL_GOALS
            )
        else:
            # Flat 模式：返回扁平状态
            return self.resource_mgr.get_flat_state(
                current_request=self.current_request,
                unadded_dest_indices=self.unadded_dest_indices,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree
            )

    # =========================================================================
    # 4. 分层动作执行（核心交互接口）
    # =========================================================================
    def step_high_level(self, goal_idx: int):
        """
        高层动作：选择下一个要接入的目标（dest_idx）

        Returns:
            state, reward, done, info
        """
        unadded_list = list(self.unadded_dest_indices)

        if goal_idx >= len(unadded_list):
            return self.get_state(), -1.0, False, {"invalid_action": True}

        # 选择目标，进入低层决策阶段
        self.current_goal_idx = goal_idx
        self.current_dest = self.current_request['dest'][unadded_list[goal_idx]]

        info = {
            "high_level_goal": self.current_dest,
            "remaining_dests": len(self.unadded_dest_indices)
        }
        return self.get_state(), 0.0, False, info

    def step_low_level(self, action: int):
        """
        低层动作：选择路径索引 i 和 k_path

        Returns:
            state, reward, sub_done（当前目标完成）, req_done（整个请求完成）, info
        """
        if self.current_request is None:
            return self.get_state(), 0.0, True, True, {"done": True}

        # 解析动作
        i_idx, k_idx = self._decode_low_level_action(action)

        network_state = self.resource_mgr.get_network_state_dict(self.current_request)

        # 计算进度（在部署前）
        if hasattr(self, 'current_goal_idx'):
            dests = self.current_request.get('dest', [])
            unadded_list = list(self.unadded_dest_indices)
            if self.current_goal_idx < len(unadded_list):
                goal_dest_idx = unadded_list[self.current_goal_idx]
                if goal_dest_idx < len(dests):
                    progress = self._compute_progress(dests[goal_dest_idx])
                else:
                    progress = 0.0
            else:
                progress = 0.0
        else:
            progress = 0.0

        # 获取部署方案
        feasible, plan, backup_used, reason = self.policy_helper.get_best_plan(
            request=self.current_request,
            network_state=network_state,
            goal_dest_idx=self.current_goal_idx if hasattr(self, 'current_goal_idx') else 0,
            k_idx=k_idx,
            i_idx=i_idx,
            current_tree=self.current_tree,
            nodes_on_tree=self.nodes_on_tree,
            path_manager=self.path_manager
        )

        reward = 0.0
        sub_done = False
        info = {"backup_used": backup_used, "feasible": feasible, "progress": progress}

        if feasible and plan is not None:
            # 成功部署
            self.resource_mgr.apply_deployment(self.current_request, plan)

            # 更新树结构
            self.current_tree['tree'] += plan.get('tree', np.zeros_like(self.current_tree['tree']))
            self.current_tree['hvt'] += plan.get('hvt', np.zeros_like(self.current_tree['hvt']))

            # 记录路径
            if hasattr(self, 'current_dest'):
                self.current_tree['paths_map'][self.current_dest] = plan.get('new_path_full', [])

            # 更新已接入节点
            new_nodes = set(np.where(plan.get('hvt', np.zeros_like(self.current_tree['hvt'])) > 0)[0])
            self.nodes_on_tree.update(new_nodes)

            # 移除已接入目标
            if hasattr(self, 'current_goal_idx'):
                unadded_list = list(self.unadded_dest_indices)
                if self.current_goal_idx < len(unadded_list):
                    self.unadded_dest_indices.discard(unadded_list[self.current_goal_idx])

            # VNF 共享记录
            for node, vnf_t in np.argwhere(plan.get('hvt', np.zeros_like(self.current_tree['hvt'])) > 0):
                self.resource_mgr.share_vnf(
                    node + 1,
                    int(vnf_t),
                    self.current_goal_idx if hasattr(self, 'current_goal_idx') else 0
                )

            # 计算奖励
            reward = self.reward_critic.criticize(
                success=True,
                backup_used=backup_used,
                progress=progress,
                sharing_rate=self.resource_mgr.get_vnf_sharing_rate()
            )

            sub_done = True
            info["success"] = True
        else:
            # 失败：可视化 + 惩罚
            if self.enable_render and hasattr(self, 'current_goal_idx'):
                self.render_failure(
                    self.current_goal_idx,
                    title=f"Fail_Req{self.current_request['id']}"
                )

            reward = self.reward_critic.criticize(
                success=False,
                backup_used=backup_used,
                progress=progress
            )
            info["reason"] = reason or "expert_failed"

        # 检查当前请求是否全部完成
        req_done = len(self.unadded_dest_indices) == 0

        if req_done:
            self.total_requests_accepted += 1
            # 注册服务以便后续离开时回收
            self.event_handler.register_service(
                self.current_request['id'],
                self.current_tree
            )

        # 时间推进与离开事件
        leaves = self.data_loader.get_current_leaves()
        self.event_handler.process_leaves(leaves)
        self.data_loader.advance_time()

        # 下一个请求
        if req_done:
            self._reset_current_request()

        return self.get_state(), reward, sub_done, req_done or self.phase_done, info

    def step(self, action):
        """兼容 gym 标准 step（仅低层）"""
        state, reward, sub_done, req_done, info = self.step_low_level(action)
        return state, reward, req_done, info
    def get_next_request_only(self):
        """
        Phase1 专用：只取请求，不初始化 rollout 状态
        """
        return self.data_loader.next_request()

    def get_network_state_snapshot(self):
        """
        Phase1 专用：给专家用的静态网络状态
        """
        return {
            "node_cap": self.resource_mgr.node_cap.copy(),
            "link_cap": self.resource_mgr.link_cap.copy(),
        }
    # =========================================================================
    # 5. 动作解码与掩码
    # =========================================================================
    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        """
        解码低层动作为 (i_idx, k_idx)

        公式: action = i_idx * K_path + k_idx
        """
        return self.policy_helper.decode_low_level_action(
            action,
            max_paths=self.path_manager.max_paths
        )

    def get_high_level_candidate_mask(self, candidates: List[Tuple[int, float]]) -> np.ndarray:
        """生成高层候选动作掩码"""
        return self.policy_helper.get_high_level_candidate_mask(
            candidates,
            self.NB_HIGH_LEVEL_GOALS
        )

    def get_low_level_action_mask(self) -> np.ndarray:
        """生成低层动作掩码"""
        return self.policy_helper.get_low_level_action_mask(
            self.path_manager,
            self.current_tree,
            self.NB_LOW_LEVEL_ACTIONS
        )

    # =========================================================================
    # 6. 专家系统包装（Phase 1/2 模仿学习 & DAgger 必需）
    # =========================================================================
    def get_expert_high_level_candidates(self, state_vec=None, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        获取专家推荐的高层候选目标

        Args:
            state_vec: 状态向量（忽略，使用 self 的属性）
            top_k: 返回前 k 个候选

        Returns:
            [(dest_idx, score), ...] 按分数降序排列
        """
        if not self.current_request or not self.unadded_dest_indices:
            return []

        network_state = self.resource_mgr.get_network_state_dict(self.current_request)

        return self.policy_helper.get_expert_candidates(
            self.current_request,
            network_state,
            self.unadded_dest_indices,
            self.current_tree,
            self.nodes_on_tree,
            top_k
        )

    def get_expert_high_level_goal(self, state_vec=None) -> int:
        """获取专家推荐的单一高层目标"""
        cands = self.get_expert_high_level_candidates(state_vec, top_k=1)
        if cands:
            return int(cands[0][0])

        # Fallback：如果专家算不出来，返回第一个未完成的
        if self.unadded_dest_indices:
            return int(next(iter(self.unadded_dest_indices)))

        return 0

    def get_expert_high_level_labels(self, state_vec=None, top_k: int = 5) -> Tuple[List[int], List[float], int]:
        """
        获取专家的高层策略标签

        Returns:
            (ids, scores, best_id): 候选ID列表、分数列表、最佳ID
        """
        cands = self.get_expert_high_level_candidates(state_vec, top_k=top_k)

        if not cands:
            return [], [], 0

        ids = [int(c[0]) for c in cands]
        scores = [float(c[1]) for c in cands]

        return ids, scores, ids[0]

    def expert_low_level_action(self, goal_dest_idx: int) -> int:
        """
        获取专家推荐的低层动作

        Returns:
            action: 专家动作索引，-1 表示无法获取
        """
        return self.policy_helper.expert_low_level_action()

    # =========================================================================
    # 7. 辅助计算方法
    # =========================================================================
    def _compute_progress(self, goal_node: int) -> float:
        """
        计算当前步进的进度值

        进度 ∈ [-1, 1]:
        - 正数：更接近目标（好）
        - 负数：离目标更远（坏）
        """
        if not self.current_request:
            return 0.0

        source_node = self.current_request.get('source', 1)

        progress = self.resource_mgr.compute_progress(
            self.nodes_on_tree,
            goal_node,
            source_node,
            self._prev_dist
        )

        # 更新上一步距离
        if self.nodes_on_tree:
            current_node = self.resource_mgr.find_closest_tree_node(
                self.nodes_on_tree, goal_node, source_node
            )
            self._prev_dist = self.resource_mgr.get_shortest_distance(current_node, goal_node)

        return progress

    def _compute_qos_violation(self) -> Optional[Dict[str, float]]:
        """计算 QoS 违规情况"""
        return self.resource_mgr.compute_qos_violation()

    def _state_novelty(self) -> float:
        """计算状态新颖度"""
        if not hasattr(self, '_state_visit_counter'):
            self._state_visit_counter = {}

        try:
            state = self.get_state()

            if state is None:
                return 0.5

            # 根据状态类型生成哈希
            if isinstance(state, tuple):  # GNN 状态
                x = state[0]

                if x is None:
                    return 0.5

                # 转换为 NumPy 数组
                if hasattr(x, 'numpy'):
                    state_array = x.flatten().numpy()
                elif hasattr(x, 'cpu'):
                    state_array = x.cpu().flatten().numpy()
                else:
                    state_array = np.array(x).flatten()

                if len(state_array) == 0:
                    return 0.5

                # 生成哈希（限制大小）
                max_features = min(100, len(state_array))
                state_hash = tuple(np.round(state_array[:max_features], 2).astype(int))

            else:  # Flat 状态
                if not isinstance(state, np.ndarray):
                    state = np.array(state)

                if len(state) == 0:
                    return 0.5

                state_hash = tuple(np.round(state, 2).astype(int))

            # 更新访问计数
            if state_hash not in self._state_visit_counter:
                self._state_visit_counter[state_hash] = 0

            self._state_visit_counter[state_hash] += 1
            visit_count = self._state_visit_counter[state_hash]

            # 计算新颖度
            novelty = 1.0 / np.sqrt(1 + visit_count)

            # 周期性清理计数器（防止内存泄漏）
            if len(self._state_visit_counter) > 10000:
                logger.warning("State visit counter exceeds 10000 entries, clearing...")
                sorted_states = sorted(
                    self._state_visit_counter.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                self._state_visit_counter = dict(sorted_states[:5000])

            return float(np.clip(novelty, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"_state_novelty error: {e}")
            return 0.5

    # =========================================================================
    # 8. 可视化与统计
    # =========================================================================
    def render_failure(self, failed_dest_idx: int, failed_path=None, title: str = "Failure"):
        """可视化部署失败的情况"""
        if not hasattr(self, 'visualizer') or not self.current_request:
            return

        if not self.enable_render:
            return

        try:
            src = self.current_request['source']
            dests = self.current_request.get('dest', [])
            failed_node = dests[failed_dest_idx] if failed_dest_idx < len(dests) else -1

            # 成功的路径
            success_paths = self.current_tree.get('paths_map', {})

            # VNF 部署
            vnf_placement = {}
            hvt = self.current_tree.get('hvt')
            if hvt is not None:
                ns, vs = np.where(hvt > 0)
                for n, v in zip(ns, vs):
                    vnf_placement[f"VNF_{v}"] = n + 1

            # 生成文件名
            req_id = self.current_request.get('id', 0)
            filename = f"fail_req{req_id}_goal{failed_dest_idx}_node{failed_node}.png"
            save_path = self.failure_output_dir / filename

            # 调用可视化器
            self.visualizer.draw_failure_case(
                src=src,
                dests=dests,
                success_paths=success_paths,
                vnf_placement=vnf_placement,
                failed_dest=failed_node,
                failed_path=failed_path,
                title=title,
                save_path=str(save_path)
            )
        except Exception as e:
            logger.debug(f"render_failure failed: {e}")

    def get_backup_metrics(self) -> Dict[str, float]:
        """获取备份策略的统计指标"""
        if not hasattr(self, 'policy_helper'):
            return {'activation_rate': 0.0, 'success_rate': 0.0}

        if not hasattr(self.policy_helper, 'backup_policy'):
            return {'activation_rate': 0.0, 'success_rate': 0.0}

        try:
            stats = self.policy_helper.backup_policy.get_statistics()
            return {
                'activation_rate': stats.get('success_rate', 0.0) * 100,
                'success_rate': stats.get('cache_hit_rate', 0.0) * 100
            }
        except:
            return {'activation_rate': 0.0, 'success_rate': 0.0}

    def print_env_summary(self):
        """打印环境统计摘要"""
        logger.info("=" * 50)
        logger.info("ENVIRONMENT SUMMARY")

        # 时间步
        logger.info(f"Time step: {self.data_loader.time_step}/{self.data_loader.total_steps}")

        # 请求统计
        seen = self.total_requests_seen
        accepted = self.total_requests_accepted
        acc_rate = accepted / max(1, seen)
        logger.info(f"Requests seen: {seen}")
        logger.info(f"Requests accepted: {accepted} ({acc_rate:.2%})")

        # 备份统计
        bk = self.get_backup_metrics()
        logger.info(f"Backup: Activated={bk['activation_rate']:.1f}%, Success={bk['success_rate']:.1f}%")

        # VNF 共享
        sharing_rate = self.resource_mgr.get_vnf_sharing_rate()
        logger.info(f"VNF Sharing Rate: {sharing_rate:.3f}")

        logger.info("=" * 50)

    def _clear_cache(self):
        """清空所有缓存"""
        self.policy_helper.clear_cache()

        if hasattr(self, '_eval_cache'):
            self._eval_cache.clear()

    # =========================================================================
    # 🔥 Phase 1 专用接口 (请添加到 SFC_HIRL_Env 类中)
    # =========================================================================
    def prepare_phase1_iterator(self):
        """Phase 1 初始化：创建一个直接遍历所有请求的迭代器"""
        if not hasattr(self.data_loader, 'requests') or not self.data_loader.requests:
            logger.error("Data loader is empty! Did you call load_dataset('phase1')?")
            self._phase1_iter = iter([])
        else:
            self._phase1_iter = iter(self.data_loader.requests)
            logger.info(f"Phase 1 iterator ready: {len(self.data_loader.requests)} requests.")

    def get_next_request_only(self):
        """Phase 1 核心：不随时间推进，直接拿下一个请求"""
        try:
            # 1. 拿下一个请求
            if not hasattr(self, '_phase1_iter'):
                self.prepare_phase1_iterator()

            req = next(self._phase1_iter)

            # 2. 手动初始化环境状态 (模拟 _reset_current_request 的部分逻辑)
            self.current_request = req
            self.total_requests_seen += 1

            # 初始化树结构
            self.nodes_on_tree = {req['source']}
            self.unadded_dest_indices = set(range(len(req['dest'])))
            self.current_tree = {
                'id': req['id'],
                'tree': np.zeros(self.resource_mgr.L, dtype=np.float32),
                'hvt': np.zeros((self.resource_mgr.n, self.resource_mgr.K_vnf), dtype=np.float32),
                'paths_map': {}
            }
            self.path_manager.reset()
            self.policy_helper.clear_cache()

            return req
        except StopIteration:
            return None

    # ==========================================
    # ✅ 修复 Attribute Error 的关键补丁
    # ==========================================
    @property
    def events(self):
        """让外部可以直接通过 env.events 访问 data_loader 里的数据"""
        return self.data_loader.events

    @property
    def requests(self):
        """让外部可以直接通过 env.requests 访问 data_loader 里的数据"""
        return self.data_loader.requests