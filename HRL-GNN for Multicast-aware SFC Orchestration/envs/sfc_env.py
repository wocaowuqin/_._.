# envs/sfc_env.py
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
import gym
import pickle
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


class SimpleTopologyManager:
    """
    简化的拓扑管理器
    如果没有独立的 TopologyManager 类，使用这个
    """

    def __init__(self, topo):
        self.topo = topo
        self.n = topo.shape[0]
        self.original_topo = topo.copy()

    def reset(self):
        """重置拓扑"""
        self.topo = self.original_topo.copy()

    def get_neighbors(self, node):
        """获取节点的邻居"""
        return np.where(self.topo[node] > 0)[0].tolist()

    def get_shortest_path(self, source, dest):
        """获取最短路径（简化版，使用 BFS）"""
        from collections import deque

        if source == dest:
            return [source]

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()

            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    new_path = path + [neighbor]

                    if neighbor == dest:
                        return new_path

                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

        return None  # 无路径


class ExpertWrapper:
    """包装 MSFCE_Solver，适配 BackupPolicy"""

    def __init__(self, msfce_solver):
        self.solver = msfce_solver
        self.node_num = msfce_solver.node_num
        self.DC = msfce_solver.DC

    def find_any_path(self, src, dst):
        """查找路径（0-based）"""
        src_1 = src + 1
        dst_1 = dst + 1
        cache_key = (src_1, dst_1, 1)

        if cache_key in self.solver._path_cache:
            nodes, dist, links = self.solver._path_cache[cache_key]
            # 转回 0-based
            nodes_0 = [n - 1 for n in nodes] if nodes else None
            return nodes_0, links
        return None, None


class SimpleDataLoader:
    """
    简化的数据加载器
    如果没有独立的 DataLoader 类，使用这个
    """

    def __init__(self, config):
        self.config = config
        self.requests = []
        self.events = []

    def load_dataset(self, phase_or_file):
        """加载数据集"""
        # 尝试从配置中获取数据路径
        if isinstance(phase_or_file, str) and phase_or_file.startswith('phase'):
            # 阶段名称
            data_dir = self.config.get('path', {}).get('input_dir', 'data/input_dir')
            req_file = os.path.join(data_dir, f'{phase_or_file}_requests.pkl')
            evt_file = os.path.join(data_dir, f'{phase_or_file}_events.pkl')
        else:
            # 直接的文件路径
            req_file = phase_or_file
            evt_file = None

        # 加载请求
        if os.path.exists(req_file):
            with open(req_file, 'rb') as f:
                self.requests = pickle.load(f)
            logger.info(f"✅ 加载请求: {len(self.requests)} 条")
        else:
            logger.warning(f"⚠️  请求文件不存在: {req_file}")
            self.requests = []

        # 加载事件
        if evt_file and os.path.exists(evt_file):
            with open(evt_file, 'rb') as f:
                self.events = pickle.load(f)
            logger.info(f"✅ 加载事件: {len(self.events)} 条")
        else:
            self.events = []

        return len(self.requests) > 0


class SFC_HIRL_Env(gym.Env):
    """
    分层强化学习 SFC 部署环境（主类）
    """

    # =========================================================================
    # 1. 初始化与配置
    # =========================================================================
    def __init__(self, config, use_gnn=False):
        """初始化环境"""
        super().__init__()

        self.config = config
        self.use_gnn = use_gnn

        # ========================================
        # 1. 加载拓扑矩阵
        # ========================================
        topo = config.get('topology', {}).get('matrix')
        if topo is None:
            n = config.get('environment', {}).get('num_nodes', 28)
            topo = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(topo, 0)
        topo = np.asarray(topo, dtype=np.float32)

        # ========================================
        # 2. 设置核心属性
        # ========================================
        self.n = topo.shape[0]
        self.K_vnf = config.get('vnf', {}).get('n_types', 8)
        self.L = int(np.sum(topo > 0))

        logger.info(f"✅ 环境参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

        # ========================================
        # 3. 初始化资源管理器
        # ========================================
        capacities = config.get('capacities', {
            'cpu': 100.0,
            'memory': 80.0,
            'bandwidth': 100.0
        })

        dc_nodes = config.get('topology', {}).get('dc_nodes', list(range(10)))
        self.dc_nodes = dc_nodes

        self.resource_mgr = ResourceManager(topo, capacities, dc_nodes)

        # ========================================
        # 4. 初始化拓扑管理器
        # ========================================
        self.topology_mgr = SimpleTopologyManager(topo)

        # ========================================
        # 5. 初始化路径管理器
        # ========================================
        self.path_manager = PathManager(max_paths=10)

        # ========================================
        # 6. 初始化 MSFCE 专家
        # ========================================
        from core.expert.expert_msfce.core.solver import MSFCE_Solver
        from core.expert.expert_msfce.utils.config import SolverConfig
        from pathlib import Path

        path_db_file = Path("data/input_dir/US_Backbone_path.mat")
        # 如果文件不存在，ExpertWrapper 可能会报错，这里假设文件存在或有处理
        msfce_solver = MSFCE_Solver(
            path_db_file=path_db_file,
            topology_matrix=topo,
            dc_nodes=dc_nodes,
            capacities=capacities,
            config=SolverConfig()
        )

        self.expert = ExpertWrapper(msfce_solver)

        # ========================================
        # 7. 初始化 BackupPolicy
        # ========================================
        from envs.modules.sfc_backup_system.backup_policy import BackupPolicy

        self.backup_policy = BackupPolicy(
            expert=self.expert,
            n=self.n,
            L=self.L,
            K_vnf=self.K_vnf,
            dc_nodes=dc_nodes
        )

        # ========================================
        # 8. 初始化数据加载器
        # ========================================
        self.data_loader = DataLoader(config)

        # ========================================
        # 9. 初始化事件处理器
        # ========================================
        self.event_handler = EventHandler(
            resource_manager=self.resource_mgr
        )

        # ========================================
        # 10. 初始化策略助手
        # ========================================
        input_dir = Path(config.get('path', {}).get('input_dir', 'data/input_dir'))

        self.policy_helper = PolicyHelper(
            input_dir=input_dir,
            topo=topo,
            dc_nodes=dc_nodes,
            capacities=capacities
        )

        # ========================================
        # 11. 初始化奖励评估器
        # ========================================
        training_phase = 3
        epoch = 0
        max_epochs = config.get('phase3', {}).get('n_episodes', 300)
        reward_params = config.get('reward', None)

        self.reward_critic = RewardCritic(
            training_phase=training_phase,
            epoch=epoch,
            max_epochs=max_epochs,
            params=reward_params
        )

        # ========================================
        # 12. 初始化失败可视化器
        # ========================================
        try:
            self.failure_visualizer = FailureVisualizer(config)
        except Exception as e:
            logger.warning(f"⚠️  FailureVisualizer 初始化失败: {e}")
            self.failure_visualizer = None

        # ========================================
        # 13. 初始化环境状态 (Critical Fixes Here)
        # ========================================
        self.phase = 'phase3'
        self.max_steps = config.get('phase3', {}).get('max_steps_per_episode', 100)
        self.step_counter = 0
        self.phase_done = False

        # ✅ 动作空间常量
        env_config = config.get('environment', {})
        self.NB_HIGH_LEVEL_GOALS = env_config.get('nb_high_level_goals', 10)

        # 🔥【修复】强制低层动作维度等于节点数，并回写配置
        self.NB_LOW_LEVEL_ACTIONS = self.n
        if 'environment' not in self.config:
            self.config['environment'] = {}
        self.config['environment']['nb_low_level_actions'] = self.n  # 回写，供 Agent 使用

        logger.info(f"✅ 动作空间锁定: 高层={self.NB_HIGH_LEVEL_GOALS}, 低层={self.NB_LOW_LEVEL_ACTIONS} (Nodes)")

        # 🔥【修复】初始化 Tree 为字典
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {}  # ✅ 必须是 Dict
        }

        # 请求相关
        self.current_request = None
        self.step_idx = 0
        self._prev_dist = None
        self.total_requests_seen = 0
        self.total_requests_accepted = 0

        self.unadded_dest_indices = set()
        self.nodes_on_tree = set()
        self.current_goal_idx = None
        self.current_dest = None

        self._state_visit_counter = {}

        # ========================================
        # 14. GNN 特征构建器
        # ========================================
        if use_gnn:
            try:
                from core.gnn.feature_builder import GNNFeatureBuilder
                self.feature_builder = GNNFeatureBuilder(config)
            except Exception as e:
                logger.warning(f"⚠️  FeatureBuilder 初始化失败: {e}")
                self.feature_builder = None
        else:
            self.feature_builder = None

        # ========================================
        # 15. Gym 空间定义
        # ========================================
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, 17), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.n, shape=(2, self.n * self.n), dtype=np.int64),
        })

        self.action_space = gym.spaces.Discrete(self.n)

        logger.info(f"✅ 环境初始化完成: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

    # =========================================================================
    # 2. 数据加载与重置
    # =========================================================================

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        """加载数据集（兼容两种调用方式）"""
        if events_file is not None:
            # 文件名方式（兼容旧逻辑）
            import pickle
            possible_dirs = [
                Path("generate_requests_depend_on_poisson/data_output"),
                self.config['path'].get('expert_data_dir', 'data/expert'),
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
                with open(req_path, 'rb') as f:
                    requests = pickle.load(f)
                with open(evt_path, 'rb') as f:
                    raw_events = pickle.load(f)

                events = []
                for evt in raw_events:
                    events.append({
                        'arrive': np.array(evt.get('arrive', []), dtype=int).flatten(),
                        'leave': np.array(evt.get('leave', []), dtype=int).flatten()
                    })

                self.data_loader.requests = requests
                self.data_loader.req_map = {r['id']: r for r in requests}
                self.data_loader.events = events
                self.data_loader.total_steps = len(events)
                self.data_loader.reset()
                logger.info(f"✓ Loaded {len(requests)} requests, {len(events)} events")
                return True
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                return False
        else:
            return self.data_loader.load_dataset(phase_or_req_file)

    def reset(self, *, seed=None, options=None):
        """Gym 标准 Reset"""
        super().reset(seed=seed)

        phase = "phase3"
        if options is not None:
            phase = options.get("phase", phase)

        logger.debug(f"[Env] reset() called, phase={phase}")

        self._reset_core()

        # 🔥【修复】Phase 3 防止重复加载数据集
        if phase == 'phase3' and len(self.requests) > 0:
            logger.debug("[Env] Phase3 requests already loaded, skipping reload.")
        else:
            if not self.load_dataset(phase):
                # 只有在真的没数据时才报错，或者尝试加载
                if len(self.requests) == 0:
                    logger.warning(f"⚠️ Failed to load dataset for {phase}, but allowing continue if intended.")
                pass

        req, _ = self.reset_request()
        obs = self.get_state()
        info = {"phase": phase}
        return obs, info

    def _reset_core(self):
        """核心重置逻辑"""
        if hasattr(self, 'resource_mgr') and self.resource_mgr is not None:
            self.resource_mgr.reset()

        if hasattr(self, 'topology_mgr') and self.topology_mgr is not None:
            self.topology_mgr.reset()

        # 🔥【修复】tree 初始化为 Dict
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {}  # ✅ Dict
        }

        self.current_request = None
        self.step_counter = 0

    def reset_request(self) -> Tuple[Optional[Dict], Any]:
        """重置并获取下一个请求"""
        self.policy_helper.clear_cache()
        self.current_request = None
        self._prev_dist = None
        self.reward_critic.on_new_request()

        while True:
            leaves = self.data_loader.get_current_leaves()
            self.event_handler.process_leaves(leaves)

            arrivals = self.data_loader.get_current_arrivals()
            if arrivals:
                self.current_request = arrivals[0]
                break

            self.data_loader.advance_time()
            if self.data_loader.is_done():
                logger.info("[Env] No more requests in dataset")
                return None, self.get_state()

        self.total_requests_seen += 1
        req = self.current_request

        dests = req.get("dest", [])
        self.unadded_dest_indices = set(range(len(dests)))
        self.nodes_on_tree = {req["source"]}

        # 🔥【修复】tree 初始化为 Dict (再次确保)
        self.current_tree = {
            "id": req["id"],
            "tree": {},  # ✅ 绝对不能是 np.zeros
            "hvt": np.zeros((self.resource_mgr.n, self.resource_mgr.K_vnf), dtype=np.float32),
            "paths_map": {}
        }

        self.path_manager.reset()
        return req, self.get_state()

    def _reset_current_request(self):
        """内部使用：获取下一个到达的请求"""
        arrivals = self.data_loader.get_current_arrivals()
        self.current_request = arrivals[0] if arrivals else None

        if self.current_request is None:
            self.phase_done = True
            return

        self.total_requests_seen += 1
        req = self.current_request

        dests = req.get('dest', [])
        self.unadded_dest_indices = set(range(len(dests)))
        self.nodes_on_tree = {req['source']}

        # 🔥【修复】tree 初始化为 Dict
        self.current_tree = {
            'tree': {},  # ✅ Dict
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'paths_map': {}
        }

        self.path_manager.reset()
        self.policy_helper.clear_cache()
        self._prev_dist = None

    # =========================================================================
    # 3. 状态获取接口
    # =========================================================================
    def get_state(self):
        """统一状态获取接口"""
        if self.use_gnn:
            return self.resource_mgr.get_graph_state(
                current_request=self.current_request,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree,
                served_dest_count=len(self.current_tree.get('paths_map', {})),
                sharing_strategy=0,
                nb_high_goals=self.NB_HIGH_LEVEL_GOALS
            )
        else:
            return self.resource_mgr.get_flat_state(
                current_request=self.current_request,
                unadded_dest_indices=self.unadded_dest_indices,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree
            )

    # =========================================================================
    # 4. 分层动作执行
    # =========================================================================
    def step_high_level(self, goal_idx: int):
        """高层动作"""
        unadded_list = list(self.unadded_dest_indices)
        if goal_idx >= len(unadded_list):
            return self.get_state(), -1.0, False, {"invalid_action": True}

        self.current_goal_idx = goal_idx
        self.current_dest = self.current_request['dest'][unadded_list[goal_idx]]

        info = {
            "high_level_goal": self.current_dest,
            "remaining_dests": len(self.unadded_dest_indices)
        }
        return self.get_state(), 0.0, False, info

    def step_low_level(self, action):
        """低层动作"""
        self.step_counter += 1

        # 🔥【修复】防御性检查
        tree_obj = self.current_tree.get('tree')
        if not isinstance(tree_obj, dict):
            logger.error(f"❌ [Fatal] current_tree['tree'] is {type(tree_obj)}, expected dict!")
            self.current_tree['tree'] = {}  # 紧急修复
            return self.get_state(), -10.0, True, False, {'error': 'tree_corrupted'}

        # 动作验证
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        if not (0 <= action < self.n):
            logger.error(f"❌ Invalid action: {action}, must be in [0, {self.n - 1}]")
            return self.get_state(), -10.0, True, False, {'error': 'invalid_action'}

        if self.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_request'}

        req = self.current_request
        vnf_types = req.get('vnf', [])
        source = req.get('source', 0)
        dests = req.get('dest', [])

        # 部署方案
        placement = {}
        for i, vnf_type in enumerate(vnf_types):
            placement[f"vnf_{i}_type_{vnf_type}"] = action

        tree = {}
        if source != action: tree[(source, action)] = 1.0
        for dest in dests:
            if dest != action: tree[(action, dest)] = 1.0

        hvt = np.zeros((self.n, self.K_vnf), dtype=np.float32)
        for i, vnf_type in enumerate(vnf_types):
            if 0 <= action < self.n and 0 <= vnf_type < self.K_vnf:
                hvt[action, vnf_type] = 1.0

        plan = {'success': True, 'placement': placement, 'tree': tree, 'hvt': hvt}

        try:
            success = self.resource_mgr.apply_tree_deployment(plan, req)
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            success = False

        if success:
            self.current_tree['hvt'] += hvt
            # 🔥【修复】安全的字典更新
            for edge_key, value in tree.items():
                self.current_tree['tree'][edge_key] = (
                        self.current_tree['tree'].get(edge_key, 0) + float(value)
                )
            self.total_requests_accepted += 1
            reward = 1.0
        else:
            reward = -1.0

        next_req, _ = self.reset_request()

        done = (next_req is None)
        truncated = (self.step_counter >= self.max_steps)

        next_state = self.get_state()
        info = {'step': self.step_counter, 'success': success}
        return next_state, reward, done, truncated, info

    def step(self, action):
        state, reward, sub_done, req_done, info = self.step_low_level(action)
        return state, reward, req_done, info

    # =========================================================================
    # 5. 辅助方法
    # =========================================================================
    def get_high_level_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.NB_HIGH_LEVEL_GOALS, dtype=np.bool_)
        if self.current_request is None:
            mask[:] = True
            return mask
        unadded = list(self.unadded_dest_indices)
        for i in range(min(len(unadded), self.NB_HIGH_LEVEL_GOALS)):
            mask[i] = True
        if not mask.any(): mask[:] = True
        return mask

    def get_low_level_action_mask(self) -> np.ndarray:
        # 简单掩码：如果还没生成树，允许所有；否则根据树生成
        # 这里简化为允许所有，或者调用 policy_helper
        if not self.current_tree.get('tree'):
            return np.ones(self.NB_LOW_LEVEL_ACTIONS, dtype=np.bool_)

        mask = self.policy_helper.get_low_level_action_mask(
            self.path_manager, self.current_tree, self.NB_LOW_LEVEL_ACTIONS
        )
        if not mask.any(): mask[:] = True
        return mask

    # Phase 1 兼容
    @property
    def events(self):
        return self.data_loader.events

    @property
    def requests(self):
        return self.data_loader.requests

    def get_next_request_only(self):
        return self.data_loader.next_request()

    # 占位符方法，避免 AttributeError
    def render_failure(self, *args, **kwargs):
        pass

    def print_env_summary(self):
        pass

    def _compute_progress(self, *args):
        return 0.0