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
from envs.modules.blacklist_manager import BlacklistManager
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
    """包装 MSFCE_Solver，适配 BackupPolicy (修复版)"""

    def __init__(self, msfce_solver):
        self.solver = msfce_solver
        # 尝试获取节点数，防错处理
        self.node_num = getattr(msfce_solver, 'node_num', 28)
        self.DC = getattr(msfce_solver, 'DC', [])

    def find_any_path(self, src, dst):
        """查找路径（0-based）"""
        # 1. 转换索引：0-based -> 1-based (适配 MATLAB/PathEngine 习惯)
        src_1 = src + 1
        dst_1 = dst + 1

        # 2. 尝试获取 PathEngine
        # 通常 MSFCE_Solver 会把 PathEngine 实例保存在 self.path_engine
        path_engine = getattr(self.solver, 'path_engine', None)

        # --- 方案 A: 通过 PathEngine 标准接口 (推荐) ---
        if path_engine and hasattr(path_engine, 'get_path_info'):
            # k=1 表示找最短路径
            nodes, dist, links = path_engine.get_path_info(src_1, dst_1, 1)
            if nodes:
                # 转回 0-based
                nodes_0 = [n - 1 for n in nodes]
                return nodes_0, links

        # --- 方案 B: 直接访问 PathEngine 缓存 (备选) ---
        if path_engine and hasattr(path_engine, '_path_cache'):
            cache_key = (src_1, dst_1, 1)
            if cache_key in path_engine._path_cache:
                nodes, dist, links = path_engine._path_cache[cache_key]
                nodes_0 = [n - 1 for n in nodes] if nodes else None
                return nodes_0, links

        # --- 方案 C: 旧版兼容 (直接在 Solver 上找) ---
        if hasattr(self.solver, '_path_cache'):
            cache_key = (src_1, dst_1, 1)
            if cache_key in self.solver._path_cache:
                nodes, dist, links = self.solver._path_cache[cache_key]
                nodes_0 = [n - 1 for n in nodes] if nodes else None
                return nodes_0, links

        # 如果都找不到
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

        # 1. 基础架构：拓扑与资源
        self._init_infrastructure()

        # 2. 核心功能模块：专家、备份、路径管理
        self._init_core_modules()

        # 3. 强化学习辅助组件：数据、奖励、策略助手
        self._init_rl_components()

        # 4. 状态与动作空间变量
        self._init_state_variables()

        # 5. GNN 与 Gym 空间定义
        self._init_gym_spaces()

        logger.info(f"✅ 环境初始化完成: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

    def _init_infrastructure(self):
        """初始化拓扑、维度和资源管理器"""
        # --- 加载拓扑 ---
        topo = self.config.get('topology', {}).get('matrix')
        if topo is None:
            n = self.config.get('environment', {}).get('num_nodes', 28)
            topo = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(topo, 0)
        self.topo = np.asarray(topo, dtype=np.float32)

        # --- 设置维度 ---
        self.n = self.topo.shape[0]
        self.K_vnf = self.config.get('vnf', {}).get('n_types', 8)
        self.L = int(np.sum(self.topo > 0))

        # --- 资源管理器 ---
        capacities = self.config.get('capacities', {'cpu': 100.0, 'memory': 80.0, 'bandwidth': 100.0})
        self.dc_nodes = self.config.get('topology', {}).get('dc_nodes', list(range(10)))

        self.resource_mgr = ResourceManager(self.topo, capacities, self.dc_nodes)
        self.topology_mgr = SimpleTopologyManager(self.topo)

        logger.info(f"✅ 环境参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

    def _init_core_modules(self):
        """初始化专家系统、备份策略和路径管理器"""
        self.path_manager = PathManager(max_paths=10)

        # --- 初始化 MSFCE 专家 ---
        try:
            from core.expert.expert_msfce.core.solver import MSFCE_Solver
            from core.expert.expert_msfce.utils.config import SolverConfig

            path_db_file = Path("data/input_dir/US_Backbone_path.mat")
            capacities = self.config.get('capacities', {})

            msfce_solver = MSFCE_Solver(
                path_db_file=path_db_file,
                topology_matrix=self.topo,
                dc_nodes=self.dc_nodes,
                capacities=capacities,
                config=SolverConfig()
            )
            self.expert = ExpertWrapper(msfce_solver)
        except ImportError as e:
            logger.error(f"❌ 无法导入专家模块: {e}")
            self.expert = None

        # --- 初始化 BackupPolicy ---
        try:
            from envs.modules.sfc_backup_system.backup_policy import BackupPolicy
            self.backup_policy = BackupPolicy(
                expert=self.expert,
                n=self.n,
                L=self.L,
                K_vnf=self.K_vnf,
                dc_nodes=self.dc_nodes
            )
        except ImportError:
            logger.warning("⚠️ 未能加载 BackupPolicy")
            self.backup_policy = None

    def _init_rl_components(self):
        """初始化数据加载、奖励计算、策略助手等"""
        self.data_loader = DataLoader(self.config)
        self.event_handler = EventHandler(resource_manager=self.resource_mgr)

        # --- Policy Helper ---
        input_dir = Path(self.config.get('path', {}).get('input_dir', 'data/input_dir'))
        capacities = self.config.get('capacities', {})
        self.policy_helper = PolicyHelper(
            input_dir=input_dir,
            topo=self.topo,
            dc_nodes=self.dc_nodes,
            capacities=capacities
        )

        # --- Reward Critic ---
        from core.reward.reward_critic import RewardCritic

        reward_params = self.config.get('reward', {})
        self.reward_critic = RewardCritic(training_phase=3, params=reward_params)
        logger.info("✅ RewardCritic已初始化")

        # --- Failure Visualizer ---
        try:
            self.failure_visualizer = FailureVisualizer(self.config)
        except Exception as e:
            logger.warning(f"⚠️ FailureVisualizer 初始化失败: {e}")
            self.failure_visualizer = None

    def _init_state_variables(self):
        """初始化环境运行时的状态变量"""
        self.phase = 'phase3'
        self.max_steps = self.config.get('phase3', {}).get('max_steps_per_episode', 200)
        self.step_counter = 0
        self.phase_done = False

        # --- 动作空间配置回写 ---
        env_config = self.config.get('environment', {})
        self.nb_high_level_goals = env_config.get('nb_high_level_goals', 10)

        # 锁定低层动作为节点数
        self.NB_LOW_LEVEL_ACTIONS = self.n
        if 'environment' not in self.config:
            self.config['environment'] = {}
        self.config['environment']['nb_low_level_actions'] = self.n

        logger.info(f"✅ 动作空间锁定: 高层={self.nb_high_level_goals}, 低层={self.NB_LOW_LEVEL_ACTIONS}")

        # --- 动态变量 ---
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {}
        }
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
        # 🔥🔥🔥【补全 1】全局活跃服务队列 (跨 Episode 存在) 🔥🔥🔥
        self.active_services = []
        # 服务驻留时间配置 (单位: Episode)
        self.service_lifetime_min = 50
        self.service_lifetime_max = 200
        self.dynamic_env = True

        blacklist_config = self.config.get('blacklist', {})
        self.blacklist_manager = BlacklistManager(
            cooldown_base=blacklist_config.get('cooldown_base', 5),
            cooldown_max=blacklist_config.get('cooldown_max', 20),
            cooldown_multiplier=blacklist_config.get('cooldown_multiplier', 2),
            cleanup_interval=blacklist_config.get('cleanup_interval', 3),
            min_valid_actions=blacklist_config.get('min_valid_actions', 3)
        )

        # 动作数量（用于掩码）
        self._n_actions = self.n

        logger.info(f"✅ 黑名单管理器已初始化 (冷却基础={blacklist_config.get('cooldown_base', 5)}步)")

    def _init_gym_spaces(self):
        """初始化 GNN 特征提取器和 Gym 空间"""
        # --- GNN Feature Builder ---
        if self.use_gnn:
            try:
                from core.gnn.feature_builder import GNNFeatureBuilder
                self.feature_builder = GNNFeatureBuilder(self.config)
            except Exception as e:
                logger.warning(f"⚠️ FeatureBuilder 初始化失败: {e}")
                self.feature_builder = None
        else:
            self.feature_builder = None

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, 17), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.n, shape=(2, self.n * self.n), dtype=np.int64),
        })
        self.action_space = gym.spaces.Discrete(self.n)
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

    def _reset_core(self):
        """核心重置逻辑 (V9.3 - 支持资源生命周期管理)"""

        # 1. 动态模式处理：先处理离开的服务，释放旧资源
        if hasattr(self, '_process_departures'):
            self._process_departures()

        # 2. 资源/拓扑重置
        if hasattr(self, 'resource_mgr') and self.resource_mgr is not None:
            # 动态环境下不能硬重置，否则正在运行的服务资源会被清空
            hard_reset = not getattr(self, 'dynamic_env', False)
            try:
                self.resource_mgr.reset(hard=hard_reset)
            except TypeError:
                self.resource_mgr.reset()

        if hasattr(self, 'topology_mgr') and self.topology_mgr is not None:
            self.topology_mgr.reset()

        self._visited_nodes = set()

        # 3. 状态变量初始化
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.current_request = None
        self.step_counter = 0
        self.current_vnf_index = 0
        self.current_node_location = 0

        # 4. 历史记录初始化
        self.last_node_location = None
        self._visited_history = []
        self.nodes_on_tree = set()

        # 计数器初始化
        import collections
        self.node_visit_counts = collections.defaultdict(int)

        # 🔥 [关键] 部署失败记忆体 (用于 Mask 驱离死循环)
        # 格式: {(request_id, vnf_index, node_id), ...}
        self.failed_deploy_attempts = set()

        # 🔥 [关键] 初始化本回合的资源分配账本 (用于失败回滚)
        # 必须在每个 Episode 开始时清空，否则回滚时会把以前的账算进来
        self.curr_ep_node_allocs = []  # 存元组: (node_id, vnf_type, cpu, mem)
        self.curr_ep_link_allocs = []  # 存元组: (u, v, bw)

        # 🔥 [关键] 确保全局活跃服务队列存在 (用于成功归档)
        if not hasattr(self, 'active_services'):
            self.active_services = []

    def reset(self, *, seed=None, options=None):
        """环境重置"""
        super().reset(seed=seed)
        phase = "phase3"
        if options:
            phase = options.get("phase", phase)

        self._reset_core()

        # ✅ 重置黑名单
        if hasattr(self, 'blacklist_manager'):
            self.blacklist_manager.reset()

        # 初始化DC节点
        self.dc_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]

        # 数据加载
        if not hasattr(self.data_loader, 'requests') or len(self.data_loader.requests) == 0:
            self.load_dataset(phase)

        # 确保按顺序重置索引
        if not hasattr(self, '_request_index'):
            self._request_index = 0

        # 重置请求
        req, obs = self.reset_request()

        if req:
            print(f"\n🔄 [RESET] Request {req.get('id')} | Src: {req.get('source')} | Dests: {req.get('dest')}")

        # 兜底
        if 'obs' not in locals():
            obs = self.get_state()

        # ✅ 返回增强的info
        info = {
            "phase": phase,
            'action_mask': self.get_action_mask(),
            'blacklist_info': self.blacklist_manager.get_info() if hasattr(self, 'blacklist_manager') else {}
        }

        return obs, info

    def reset_request(self) -> Tuple[Optional[Dict], Any]:
        """重置请求 (确保清空账本)"""
        self.policy_helper.clear_cache()
        self.current_request = None
        self._prev_dist = None
        if hasattr(self, 'reward_critic'):
            self.reward_critic.on_new_request()

        if not hasattr(self, '_request_index'): self._request_index = 0
        if not hasattr(self.data_loader, 'requests') or not self.data_loader.requests:
            return None, self.get_state()

        if self._request_index >= len(self.data_loader.requests):
            self._request_index = 0

        raw_req = self.data_loader.requests[self._request_index]
        self._request_index += 1
        self.total_requests_seen += 1

        req = raw_req.copy()
        src = req.get("source", 0)
        if src >= self.n: src -= 1
        req['source'] = src

        new_dests = [d - 1 if d >= self.n else d for d in req.get("dest", [])]
        req['dest'] = new_dests

        new_vnfs = [v - 1 if v >= self.K_vnf else v for v in req.get('vnf', [])]
        req['vnf'] = new_vnfs

        self.current_request = req
        self.unadded_dest_indices = set(range(len(new_dests)))
        self.current_node_location = req['source']
        self.nodes_on_tree = {req['source']}
        self.current_vnf_index = 0

        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }

        # ============================================
        # 🔥 [关键修复 6] 每次请求必须清空账本！
        # ============================================
        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []

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
        self.current_node_location = req['source']
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
        # envs/sfc_env.py

    def get_state(self):
        """统一状态获取接口 (GNN模式 + 真实 One-Hot 编码填充)"""
        if self.use_gnn:
            # 1. 获取基础图数据
            raw = self.resource_mgr.get_graph_state(
                current_request=self.current_request,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree,
                served_dest_count=len(self.current_tree.get('tree', {})),
                sharing_strategy=0,
                nb_high_goals=self.nb_high_level_goals
            )

            # 2. 转换为 PyG Data 并填充 Req Vec
            try:
                from torch_geometric.data import Data
                import torch

                data = Data(**raw) if isinstance(raw, dict) else (
                    Data(x=raw[0], edge_index=raw[1]) if isinstance(raw, tuple) else raw)

                # 🔥 One-Hot 修复逻辑
                target_dim = 24
                real_vec = torch.zeros((1, target_dim), dtype=torch.float32)
                if self.current_request:
                    vnf_chain = self.current_request.get('vnf', [])
                    max_len = target_dim // self.K_vnf
                    for i, vnf_type in enumerate(vnf_chain[:max_len]):
                        idx = i * self.K_vnf + vnf_type
                        if idx < target_dim: real_vec[0, idx] = 1.0

                data.req_vec = real_vec
                return data
            except:
                return raw
        else:
            return np.zeros(32)
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

    def _get_action_metrics(self, node_action: int, tree_edges: Dict) -> Tuple[float, float, int]:
        """从资源管理器提取指标 (修复首节点带宽 Bug)"""
        try:
            cpu_remain = float(self.resource_mgr.C[node_action])
        except (IndexError, AttributeError) as e:
            cpu_remain = 0.0

        min_bw = 99999.0
        hops = 0

        if tree_edges:
            hops = len(tree_edges)
            for edge_key, _ in tree_edges.items():
                u, v = None, None
                if isinstance(edge_key, tuple):
                    u, v = edge_key
                elif isinstance(edge_key, str):
                    try:
                        u, v = map(int, edge_key.strip('()').split('-'))
                    except ValueError:
                        continue

                if u is not None and v is not None:
                    if (u, v) in self.resource_mgr.links['bandwidth']:
                        bw = self.resource_mgr.links['bandwidth'][(u, v)]
                        if bw < min_bw: min_bw = float(bw)

            # 如果遍历完还是初始值，说明没有有效边
            if min_bw > 90000.0: min_bw = 0.0
        else:
            # 🔥 修复：第一个节点没有边，带宽视为满额 (100.0)，避免被 RewardCritic 误杀
            min_bw = 80.0

        return cpu_remain, min_bw, hops

    # 在step方法中调用
    def step(self, action):
        """
        执行动作 (完整替换)

        主要修改：
        1. 增加步数计数（黑名单）
        2. 部署失败时加入黑名单
        3. 返回增强的info（包含action_mask和blacklist_info）
        """
        self.step_counter += 1

        # ✅ 增加黑名单步数计数
        if hasattr(self, 'blacklist_manager'):
            self.blacklist_manager.increment_step()

        done = False
        truncated = False
        reward = 0.0
        info = {'success': False}

        # --- 1. 基础检查 ---
        if self.current_request is None:
            return self.get_state(), -1.0, True, False, info

        target_node = int(action)
        current_node = self.current_node_location

        dests = self.current_request.get('dest', [])
        vnf_list = self.current_request.get('vnf', [])

        # 更新访问计数
        if not hasattr(self, 'node_visit_counts'):
            import collections
            self.node_visit_counts = collections.defaultdict(int)
        self.node_visit_counts[target_node] += 1
        visit_count = self.node_visit_counts[target_node]

        # --- 2. 执行阶段逻辑 ---
        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        # ============================================
        # 🚀 阶段A：VNF部署
        # ============================================
        if not is_vnf_complete:
            info['phase'] = 'vnf_deployment'
            if target_node == current_node:
                info['action_type'] = 'deploy'
                # 尝试部署
                deploy_success = self._try_deploy(current_node)

                new_deployed = len(self.current_tree.get('placement', {}))
                all_complete = (new_deployed >= len(vnf_list))
                reward = self.reward_critic.compute_vnf_deploy_reward(deploy_success, all_complete)

                if deploy_success:
                    info['success'] = True
                    if all_complete:
                        print(f"🎉 所有VNF部署完成！进入树构建阶段")
                        self.node_visit_counts.clear()
                        self.node_visit_counts[current_node] = 1
                else:
                    # ✅ 部署失败 → 加入黑名单
                    info['error'] = 'deploy_failed'
                    info['message'] = f"资源不足 (节点{current_node} VNF{deployed_count})"

                    if hasattr(self, 'blacklist_manager'):
                        self.blacklist_manager.add_node(current_node, "资源不足")

            else:
                info['action_type'] = 'move_in_deployment'
                valid_link = self._check_link_validity(current_node, target_node)
                to_dc = (target_node in getattr(self, 'dc_nodes', []))
                reward = self.reward_critic.compute_vnf_move_reward(to_dc, valid_link)

                if valid_link:
                    self.current_node_location = target_node
                    self.nodes_on_tree.add(target_node)
                else:
                    info['error'] = 'invalid_link'

        # ============================================
        # 🌳 阶段B：树构建
        # ============================================
        else:
            info['phase'] = 'tree_building'
            if 'connected_dests' not in self.current_tree:
                self.current_tree['connected_dests'] = set()
            connected = self.current_tree['connected_dests']
            unconnected = [d for d in dests if d not in connected]
            total_dests = len(dests)

            vnf_hub = -1
            if self.current_tree.get('placement'):
                vnf_hub = self.current_tree['placement'].get(0, -1)
            is_hub = (target_node == vnf_hub)

            if target_node == current_node:
                # 原地连接逻辑
                if current_node in unconnected:
                    connected.add(current_node)
                    conn_count = len(connected)
                    is_complete = (conn_count >= total_dests)
                    reward = self.reward_critic.compute_tree_connection_reward(conn_count, total_dests, is_complete)
                    reward += self.reward_critic.compute_frequency_penalty(visit_count, is_hub)
                    print(f"🎯 连接成功: 节点{current_node} ({conn_count}/{total_dests})")

                    if is_complete:
                        done = True
                        info['request_completed'] = True
                        self._archive_request(success=True)
                        print(f"🎉🎉 完美完成！")
                else:
                    reward = self.reward_critic.params.wrong_position
            else:
                # 移动逻辑
                info['action_type'] = 'move_in_tree'
                valid_link = self._check_link_validity(current_node, target_node)

                min_dist_before = self._min_distance_to_unconnected(current_node, unconnected)
                min_dist_after = self._min_distance_to_unconnected(target_node, unconnected)
                to_dest = (target_node in unconnected)

                reward = self.reward_critic.compute_tree_move_reward(to_dest, valid_link, min_dist_before,
                                                                     min_dist_after)
                reward += self.reward_critic.compute_frequency_penalty(visit_count, is_hub)

                if valid_link:
                    # 树边资源分配
                    edge_key = (current_node, target_node)
                    if 'tree' not in self.current_tree:
                        self.current_tree['tree'] = {}

                    # 只有新边才扣费
                    if edge_key not in self.current_tree['tree']:
                        bw_demand = self.current_request.get('bandwidth', 1.0)
                        try:
                            if self.resource_mgr.allocate_link_resource(current_node, target_node, bw_demand):
                                self.curr_ep_link_allocs.append((current_node, target_node, bw_demand))
                                print(f"📝 记账边: {current_node}->{target_node} BW={bw_demand:.1f}")
                        except Exception as e:
                            print(f"❌ 分配异常: {e}")

                    # 更新树拓扑
                    self._update_tree_state(current_node, target_node)
                    self.current_node_location = target_node

                    # 自动连接逻辑
                    if target_node in unconnected:
                        connected.add(target_node)
                        conn_count = len(connected)
                        is_complete = (conn_count >= total_dests)

                        reward += self.reward_critic.compute_tree_connection_reward(conn_count, total_dests,
                                                                                    is_complete)
                        print(f"🎯 移动并连接: 节点{target_node} ({conn_count}/{total_dests})")

                        if is_complete:
                            done = True
                            info['request_completed'] = True
                            self._archive_request(success=True)
                            print(f"🎉🎉 完美完成！")

        # --- 3. 超时处理 ---
        if self.step_counter >= self.max_steps:
            truncated = True
            done = True
            in_vnf_phase = not is_vnf_complete
            conn_count = len(self.current_tree.get('connected_dests', []))
            timeout_reward = self.reward_critic.compute_timeout_reward(in_vnf_phase, conn_count, len(dests))
            reward += timeout_reward

            if not in_vnf_phase:
                ratio = conn_count / len(dests) if dests else 0
                if ratio >= 0.8:
                    print(f"⏰✅ 超时但高进度 ({ratio:.1%}) -> 视为成功")
                    self._archive_request(success=True)
                else:
                    print(f"⏰❌ 超时失败 ({ratio:.1%}) -> 触发回滚")
                    self._archive_request(success=False)
            else:
                print(f"⏰❌ VNF阶段超时 -> 触发回滚")
                self._archive_request(success=False)

        # ✅ 返回增强的info
        if hasattr(self, 'blacklist_manager'):
            info['action_mask'] = self.get_action_mask()
            info['blacklist_info'] = self.blacklist_manager.get_info()
            info['step'] = self.blacklist_manager._step_counter

        return self.get_state(), reward, done, truncated, info
    # =========================================================================
    # 5. 辅助方法
    # =========================================================================
    def get_high_level_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.nb_high_level_goals, dtype=np.bool_)
        if self.current_request is None:
            mask[:] = True
            return mask
        unadded = list(self.unadded_dest_indices)
        for i in range(min(len(unadded), self.nb_high_level_goals)):
            mask[i] = True
        if not mask.any(): mask[:] = True
        return mask

        # =========================================================================
        # 1. 核心逻辑：Step & Mask
        # =========================================================================

    def _is_tree_building_terminal(self):
        """
        [辅助方法] 判断是否处于树构建的终止阶段
        定义：只剩 1 个或 0 个未连接的目的节点。
        作用：在此阶段放宽所有限制（如 Visit Count），确保必须连通。
        """
        if not self.current_request:
            return False

        # 获取连接状态
        connected = self.current_tree.get('connected_dests', set())
        all_dests = self.current_request.get('dest', [])

        # 剩余节点数
        remaining = len(all_dests) - len(connected)

        return remaining <= 1

    def step_low_level(self, action):
        """
        🔥 [V9.8 逻辑修正版] Step 逻辑
        修复：
        1. VNF阶段移动只更新位置，不记录树边（防止污染拓扑）
        2. 获取 vnf_hub 时使用整数索引 0
        3. 详细的带宽记账日志
        """
        self.step_counter += 1
        reward = 0.0
        done = False
        truncated = False

        target_node = int(action)
        current_node = self.current_node_location

        info = {
            'action_type': 'unknown',
            'success': False,
            'phase': None
        }

        # --- 0. 基础合法性检查 ---
        if target_node < 0 or target_node >= self.n:
            return self.get_state(), -10.0, True, False, {'error': 'invalid_range'}
        if self.current_request is None:
            return self.get_state(), -5.0, True, False, {'error': 'no_request'}

        req = self.current_request
        vnf_list = req.get('vnf', [])
        dests = req.get('dest', [])

        # Mask 检查
        mask = self.get_low_level_action_mask()
        if not mask[target_node]:
            self._diagnose_illegal_action(current_node, target_node, vnf_list, dests)
            self._archive_request(success=False)  # 立即归档触发回滚
            reward = self.reward_critic.get_reward(phase='penalty', type='invalid_action')
            return self.get_state(), reward, True, False, {'error': 'illegal_action'}

        # 更新访问计数
        if not hasattr(self, 'node_visit_counts'):
            import collections
            self.node_visit_counts = collections.defaultdict(int)
        self.node_visit_counts[target_node] += 1
        visit_count = self.node_visit_counts[target_node]

        # --- 2. 执行阶段逻辑 ---
        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        # ============================================
        # 🚀 阶段A：VNF部署
        # ============================================
        if not is_vnf_complete:
            info['phase'] = 'vnf_deployment'
            if target_node == current_node:
                info['action_type'] = 'deploy'
                # 内部调用 _try_deploy (已包含记账)
                deploy_success = self._try_deploy(current_node)

                new_deployed = len(self.current_tree.get('placement', {}))
                all_complete = (new_deployed >= len(vnf_list))
                reward = self.reward_critic.compute_vnf_deploy_reward(deploy_success, all_complete)

                if deploy_success:
                    info['success'] = True
                    if all_complete:
                        print(f"🎉 所有VNF部署完成！进入树构建阶段")
                        self.node_visit_counts.clear()
                        self.node_visit_counts[current_node] = 1
                else:
                    info['error'] = 'deploy_failed'
            else:
                info['action_type'] = 'move_in_deployment'
                valid_link = self._check_link_validity(current_node, target_node)
                to_dc = (target_node in getattr(self, 'dc_nodes', []))
                reward = self.reward_critic.compute_vnf_move_reward(to_dc, valid_link)

                if valid_link:
                    # ✅ [关键修复] VNF阶段移动：只更新位置，绝不记录树边！
                    self.current_node_location = target_node

                    # 仅仅追踪一下访问过的点，方便画图，但不作为树的边
                    self.nodes_on_tree.add(target_node)

                    # ❌ 禁止调用 self._update_tree_state(current_node, target_node)
                else:
                    info['error'] = 'invalid_link'

        # ============================================
        # 🌳 阶段B：树构建
        # ============================================
        else:
            info['phase'] = 'tree_building'
            if 'connected_dests' not in self.current_tree:
                self.current_tree['connected_dests'] = set()
            connected = self.current_tree['connected_dests']
            unconnected = [d for d in dests if d not in connected]
            total_dests = len(dests)

            # ✅ [修复] 使用整数索引 0 获取 Hub
            vnf_hub = -1
            if self.current_tree.get('placement'):
                vnf_hub = self.current_tree['placement'].get(0, -1)
            is_hub = (target_node == vnf_hub)

            if target_node == current_node:
                # 原地连接逻辑
                if current_node in unconnected:
                    connected.add(current_node)
                    conn_count = len(connected)
                    is_complete = (conn_count >= total_dests)
                    reward = self.reward_critic.compute_tree_connection_reward(conn_count, total_dests, is_complete)
                    reward += self.reward_critic.compute_frequency_penalty(visit_count, is_hub)
                    print(f"🎯 连接成功: 节点{current_node} ({conn_count}/{total_dests})")

                    if is_complete:
                        done = True
                        info['request_completed'] = True
                        self._archive_request(success=True)
                        print(f"🎉🎉 完美完成！")
                else:
                    reward = self.reward_critic.params.wrong_position
            else:
                # 移动逻辑
                info['action_type'] = 'move_in_tree'
                valid_link = self._check_link_validity(current_node, target_node)

                min_dist_before = self._min_distance_to_unconnected(current_node, unconnected)
                min_dist_after = self._min_distance_to_unconnected(target_node, unconnected)
                to_dest = (target_node in unconnected)

                reward = self.reward_critic.compute_tree_move_reward(to_dest, valid_link, min_dist_before,
                                                                     min_dist_after)
                reward += self.reward_critic.compute_frequency_penalty(visit_count, is_hub)

                if valid_link:
                    # ✅ [记账] 树边资源分配
                    edge_key = (current_node, target_node)
                    if 'tree' not in self.current_tree: self.current_tree['tree'] = {}

                    # 只有新边才扣费
                    if edge_key not in self.current_tree['tree']:
                        bw_demand = self.current_request.get('bandwidth', 1.0)
                        try:
                            if self.resource_mgr.allocate_link_resource(current_node, target_node, bw_demand):
                                self.curr_ep_link_allocs.append((current_node, target_node, bw_demand))
                                print(
                                    f"📝 记账边: {current_node}->{target_node} BW={bw_demand:.1f} (总计{len(self.curr_ep_link_allocs)}条)")
                            else:
                                pass  # 带宽不足暂时放行
                        except Exception as e:
                            print(f"❌ 分配异常: {e}")

                    # 更新树拓扑 (这里调用update是合法的)
                    self._update_tree_state(current_node, target_node)
                    self.current_node_location = target_node

                    # 自动连接逻辑
                    if target_node in unconnected:
                        connected.add(target_node)
                        conn_count = len(connected)
                        is_complete = (conn_count >= total_dests)

                        reward += self.reward_critic.compute_tree_connection_reward(conn_count, total_dests,
                                                                                    is_complete)
                        print(f"🎯 移动并连接: 节点{target_node} ({conn_count}/{total_dests})")

                        if is_complete:
                            done = True
                            info['request_completed'] = True
                            self._archive_request(success=True)
                            print(f"🎉🎉 完美完成！")

        # --- 3. 超时处理 ---
        if self.step_counter >= self.max_steps:
            truncated = True
            done = True
            in_vnf_phase = not is_vnf_complete
            conn_count = len(self.current_tree.get('connected_dests', []))
            timeout_reward = self.reward_critic.compute_timeout_reward(in_vnf_phase, conn_count, len(dests))
            reward += timeout_reward

            if not in_vnf_phase:
                ratio = conn_count / len(dests) if dests else 0
                if ratio >= 0.8:
                    print(f"⏰✅ 超时但高进度 ({ratio:.1%}) -> 视为成功")
                    self._archive_request(success=True)
                else:
                    print(f"⏰❌ 超时失败 ({ratio:.1%}) -> 触发回滚")
                    self._archive_request(success=False)
            else:
                print(f"⏰❌ VNF阶段超时 -> 触发回滚")
                self._archive_request(success=False)

        return self.get_state(), reward, done, truncated, info

    def get_low_level_action_mask(self):
        """
        🔥 V11.0 软惩罚版
        Mask策略：极度宽松，只禁止明显错误的动作
        """
        mask = np.zeros(self.n, dtype=np.bool_)
        curr = self.current_node_location

        if self.current_request is None:
            return mask

        # 获取物理邻居
        neighbors = []
        try:
            if hasattr(self, 'resource_mgr'):
                neighbors = self.resource_mgr.get_neighbors(curr)
            else:
                neighbors = np.where(self.topo[curr] > 0)[0].tolist()
        except:
            neighbors = []

        # 判断阶段
        vnf_list = self.current_request.get('vnf', [])
        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        if not is_vnf_complete:
            # ============================================
            # VNF阶段：允许所有移动
            # ============================================
            for neighbor in neighbors:
                mask[neighbor] = 1

            # 检查当前节点是否可以部署
            can_deploy_here = False
            if curr in getattr(self, 'dc_nodes', []):
                if self._check_deployment_validity(curr):
                    can_deploy_here = True

            mask[curr] = 1 if can_deploy_here else 0

        else:
            # ============================================
            # 树构建阶段：极度宽松，只禁止重复边
            # ============================================
            if 'connected_dests' not in self.current_tree:
                self.current_tree['connected_dests'] = set()

            connected = self.current_tree['connected_dests']
            all_dests = self.current_request.get('dest', [])
            unconnected = [d for d in all_dests if d not in connected]

            # 如果当前位置是未连接的目的节点，允许原地连接
            if curr in unconnected:
                mask[curr] = 1
            else:
                mask[curr] = 0

            # 获取已有边
            existing_edges = set()
            if 'tree' in self.current_tree:
                existing_edges = set(self.current_tree['tree'].keys())

            # 遍历邻居
            for neighbor in neighbors:
                # ============================================
                # 🔥 只禁止两种情况
                # ============================================

                # 1. 重复边（同一条边走两次没意义）
                if (curr, neighbor) in existing_edges:
                    continue

                # 2. 不可达任何目标（物理上不连通）
                if neighbor not in unconnected:
                    can_reach = False
                    for dest in unconnected:
                        try:
                            path = self.topology_mgr.get_shortest_path(neighbor, dest)
                            if path:
                                can_reach = True
                                break
                        except:
                            can_reach = True  # 乐观假设
                            break

                    if not can_reach:
                        continue

                # ============================================
                # ✅ 允许这个动作（即使是反向边、频繁访问）
                # ============================================
                mask[neighbor] = 1

            # Fallback（极少触发）
            if mask.sum() == 0:
                print(f"⚠️ Mask为空，允许所有邻居")
                for neighbor in neighbors:
                    mask[neighbor] = 1

        # 最终兜底
        if mask.sum() == 0:
            mask[curr] = 1

        return mask
    def _get_current_bandwidth_need(self):
        """获取当前步骤所需的带宽"""
        if not self.current_request:
            return 0.0

        # 简单逻辑：假设所有链路带宽需求一致，取请求中的第一个值
        # 或者根据当前 VNF 阶段获取特定带宽
        bw_reqs = self.current_request.get('bandwidth', [])
        if isinstance(bw_reqs, list) and len(bw_reqs) > 0:
            return float(bw_reqs[0])
        return 1.0  # 默认值

    def _archive_request(self, success: bool):
        """
        归档请求（成功或失败）

        修改：服务释放时清理黑名单
        """
        # ... 原有逻辑保持不变 ...

        if not success:
            # 失败：回滚资源
            self._rollback_resources()
            logger.info(f"❌ 请求{self.current_request.get('id')}失败，资源已回滚")
        else:
            # 成功：归档服务
            req_id = self.current_request.get('id', -1)

            # 生成TTL
            import random
            ttl = random.randint(
                getattr(self, 'service_lifetime_min', 50),
                getattr(self, 'service_lifetime_max', 200)
            )

            service_record = {
                'id': req_id,
                'ttl_remaining': ttl,
                'node_allocs': self.curr_ep_node_allocs.copy(),
                'link_allocs': self.curr_ep_link_allocs.copy()
            }

            if not hasattr(self, 'active_services'):
                self.active_services = []
            self.active_services.append(service_record)

            logger.info(f"✅ 请求{req_id}成功，服务已归档 (TTL={ttl})")
            self.total_requests_accepted += 1

    def _should_deploy_at_current_node(self) -> bool:
        """
        判断是否应该在当前位置部署 VNF
        """
        if self.current_request is None:
            return False

        current_node = getattr(self, 'current_node_location',
                               self.current_request.get('source', 0))

        # 1. 位置合法性：必须是 DC 节点
        if hasattr(self, 'dc_nodes') and current_node not in self.dc_nodes:
            return False

        # 2. 任务状态：是否还有 VNF 待部署
        vnf_types = self.current_request.get('vnf', [])
        deployed_count = len(self.current_tree.get('placement', {}))
        if deployed_count >= len(vnf_types):
            return False  # 任务已完成

        # 3. 资源检查：当前节点 CPU 是否足够下一个 VNF
        try:
            cpu_needed = self.current_request.get('cpu_origin', [])[deployed_count]
            if self.resource_mgr.C[current_node] < cpu_needed:
                return False  # 资源不足
        except Exception:
            return False

        return True

    def _execute_deployment(self, deployment_node: int):
        """执行 VNF 部署"""
        req = self.current_request
        vnf_types = req.get('vnf', [])
        source = req.get('source', 0)
        dests = req.get('dest', [])
        deployed_count = len(self.current_tree.get('placement', {}))

        # 1. 防御性检查
        if deployed_count >= len(vnf_types):
            return self.get_state(), -1.0, False, False, {'error': 'already_done'}

        vnf_type = vnf_types[deployed_count]

        # 2. 尝试资源部署 (调用 ResourceManager)
        # 构造临时的 plan 对象传给 resource_mgr
        placement = {f"vnf_{deployed_count}_type_{vnf_type}": deployment_node}

        # 简单的 Tree 结构 (Source -> Current -> Dests)
        # 注意：这里的 Tree 只是为了计费，不是真实路径
        tree = {}
        if source != deployment_node: tree[(source, deployment_node)] = 1.0
        for d in dests:
            if d != deployment_node: tree[(deployment_node, d)] = 1.0

        hvt = np.zeros((self.n, self.K_vnf), dtype=np.float32)
        hvt[deployment_node, vnf_type] = 1.0

        plan = {'success': True, 'placement': placement, 'tree': tree, 'hvt': hvt}

        success = False
        try:
            success = self.resource_mgr.apply_tree_deployment(plan, req)
        except Exception:
            success = False

        # 3. 更新状态
        if success:
            self.current_tree['hvt'] += hvt
            # 累加 tree 负载
            for k, v in tree.items():
                self.current_tree['tree'][k] = self.current_tree['tree'].get(k, 0) + v

            if 'placement' not in self.current_tree: self.current_tree['placement'] = {}
            self.current_tree['placement'].update(placement)

            self.total_requests_accepted += 1

        # 4. 奖励计算 (调用 RewardCritic)
        cpu_val, bw_val, hops_val = 0.0, 0.0, 0
        if success:
            cpu_val, bw_val, hops_val = self._get_action_metrics(deployment_node, tree)

        is_last_vnf = (deployed_count + 1 == len(vnf_types))
        request_completed = success and is_last_vnf

        if not success:
            reward = self.reward_critic.criticize(request_failed=True)
        else:
            reward = self.reward_critic.criticize(
                request_completed=request_completed,
                sub_task_completed=True,
                cpu_remain=cpu_val, bandwidth=bw_val, hops=hops_val,
                is_meta_step=True
            )

        # 5. 流程控制
        # 如果请求全部完成 OR 部署失败，则结束当前 Request，读取下一个
        done = False
        next_req = None

        if request_completed or not success:
            next_req, _ = self.reset_request()
            # 如果没有下一个请求了，则 Episode 结束
            if next_req is None: done = True

        truncated = (self.step_counter >= self.max_steps)

        info = {
            'step': self.step_counter,
            'success': success,
            'action_type': 'deploy',
            'node': deployment_node,
            'all_deployed': request_completed
        }

        return self.get_state(), reward, done, truncated, info
    def _execute_movement(self, current_node: int, target_node: int):
        """执行物理移动"""
        # 1. 物理拓扑检查
        if hasattr(self, 'topology_mgr'):
            neighbors = self.topology_mgr.get_neighbors(current_node)
            neighbors.append(current_node)
        else:
            neighbors = list(range(self.n))

        if target_node not in neighbors:
            # 瞬移惩罚
            info = {'error': 'teleportation', 'from': current_node, 'to': target_node}
            return self.get_state(), -2.0, False, False, info

        # 2. 更新物理位置
        self.current_node_location = target_node
        # 记录路径（防止画圈，用于 Feature Builder）
        self.nodes_on_tree.add(target_node)

        # 3. 计算移动奖励
        # 移动本身有成本（-0.1），如果是原地不动（等待）且没资源，可能惩罚更多
        reward = -0.1
        if target_node == current_node:
            reward = -0.2  # 鼓励移动而不是发呆

        # 4. 状态更新
        # 🔥🔥🔥 关键修复：移动并未完成任务，不要 Reset Request！🔥🔥🔥
        done = False
        truncated = (self.step_counter >= self.max_steps)
        next_state = self.get_state()

        # 检查是否刚到达了一个可部署点（给点小甜头？）
        # reached_deployable = self._should_deploy_at_current_node()
        # if reached_deployable: reward += 0.5

        info = {
            'step': self.step_counter,
            'action_type': 'move',
            'from': current_node,
            'to': target_node
        }

        return next_state, reward, done, truncated, info
    def _check_node_resource(self, node: int) -> bool:
        """检查节点资源是否足够（用于 Mask 预判）"""
        try:
            if self.current_request is None: return True
            vnf_types = self.current_request.get('vnf', [])
            deployed = len(self.current_tree.get('placement', {}))
            if deployed < len(vnf_types):
                cpu = self.current_request.get('cpu_origin', [])[deployed]
                return self.resource_mgr.C[node] >= cpu
        except:
            pass
        return True

    def set_dynamic_mode(self, enabled: bool):
        """由 Trainer 调用，控制是否开启 TTL 离去机制"""
        self.dynamic_env = enabled
        # logger.info(f"🔄 环境动态模式已切换为: {enabled}")

    def _process_departures(self):
        """
        处理服务离开（动态环境）

        修改：服务释放时清理黑名单
        """
        if not hasattr(self, 'active_services'):
            self.active_services = []

        if not self.active_services:
            return

        # 减少所有服务的TTL
        for svc in self.active_services:
            svc['ttl_remaining'] -= 1

        # 收集需要释放的服务
        to_release = [svc for svc in self.active_services if svc['ttl_remaining'] <= 0]

        if not to_release:
            return

        # ✅ 收集释放的节点
        released_nodes = set()

        # 释放资源
        for svc in to_release:
            # 释放节点资源
            for node_id, vnf_type, cpu, mem in svc.get('node_allocs', []):
                if hasattr(self, 'resource_mgr'):
                    self.resource_mgr.release_node_resource(node_id, cpu, mem)
                released_nodes.add(node_id)

            # 释放链路资源
            for u, v, bw in svc.get('link_allocs', []):
                if hasattr(self, 'resource_mgr'):
                    self.resource_mgr.release_link_resource(u, v, bw)

            # 从活跃列表移除
            self.active_services.remove(svc)
            logger.debug(f"👋 服务{svc['id']}已释放")

        # ✅ 清理黑名单
        if hasattr(self, 'blacklist_manager') and released_nodes:
            self.blacklist_manager.on_service_release(
                released_nodes,
                resource_checker=self._check_node_resources
            )

        print(f"👋 [资源释放完成] 本轮释放了 {len(to_release)} 个服务, 剩余 {len(self.active_services)} 个")

    def _rollback_current_episode(self):
        """
        🔥 [V9.8 可视化版] 回滚当前失败的回合
        """
        if not self.curr_ep_node_allocs and not self.curr_ep_link_allocs:
            return

        print(f"🔄 开始回滚: {len(self.curr_ep_node_allocs)}个VNF + {len(self.curr_ep_link_allocs)}条边")

        # 1. 回滚节点
        for (n, vt, c, m) in self.curr_ep_node_allocs:
            self.resource_mgr.release_node_resource(n, vt, c, m)
            print(f"   ↩️  释放节点{n}: VNF{vt} CPU={c:.1f}")

        # 2. 回滚链路
        for (u, v, bw) in self.curr_ep_link_allocs:
            self.resource_mgr.release_link_resource(u, v, bw)
            print(f"   ↩️  释放边: {u}->{v} BW={bw:.1f}")

        print(f"✅ 回滚完成，资源已释放")

        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []

    def _check_deployment_validity(self, node_id):
        """部署资格检查 (修复DC列表)"""
        if self.current_request is None: return False

        # 确保DC列表正确
        if not hasattr(self, 'dc_nodes'):
            self.dc_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]

        # 去重并排序，防止打印 233 这种错误
        self.dc_nodes = sorted(list(set([int(x) for x in self.dc_nodes if x < self.n])))

        if node_id not in self.dc_nodes: return False

        source = self.current_request.get('source', -1)
        if node_id == source: return False

        dests = self.current_request.get('dest', [])
        if node_id in dests: return False

        return True

    def _try_deploy(self, node_id):
        """
        🔥 [V9.8 整数索引版] VNF部署
        修复：
        1. 使用整数 idx 作为 placement 的 key
        2. 添加详细记账日志
        """
        try:
            if self.current_request is None: return False
            req = self.current_request
            vnf_list = req.get('vnf', [])

            if 'placement' not in self.current_tree:
                self.current_tree['placement'] = {}

            # ✅ [修复] 使用整数索引，而非字符串
            idx = len(self.current_tree['placement'])
            if idx >= len(vnf_list): return False

            vnf_type = vnf_list[idx]

            # 1. 资格检查
            if not self._check_deployment_validity(node_id):
                return False

            # 2. 获取资源需求
            cpu_reqs = req.get('cpu_origin', [])
            mem_reqs = req.get('memory_origin', [])
            c_need = cpu_reqs[idx] if idx < len(cpu_reqs) else 1.0
            m_need = mem_reqs[idx] if idx < len(mem_reqs) else 1.0

            # 3. 尝试扣除资源
            success = self.resource_mgr.allocate_node_resource(node_id, vnf_type, c_need, m_need)

            if not success:
                req_id = req.get('id', -1)
                fail_key = (req_id, idx, node_id)
                self.failed_deploy_attempts.add(fail_key)
                print(f"❌ [资源不足] 节点{node_id} 无法部署 VNF#{idx}")
                return False

            # ====================================================
            # 🔥 [记账] 记录VNF资源分配
            # ====================================================
            self.curr_ep_node_allocs.append((node_id, vnf_type, c_need, m_need))
            print(f"📝 记账VNF: 节点{node_id} CPU={c_need:.1f} (总计{len(self.curr_ep_node_allocs)}项)")

            # 4. 成功：记录状态
            # ✅ [修复] 统一用整数 Key
            self.current_tree['placement'][idx] = node_id

            if 'hvt' not in self.current_tree:
                self.current_tree['hvt'] = np.zeros((self.n, self.K_vnf), dtype=np.float32)
            self.current_tree['hvt'][node_id, vnf_type] += 1.0

            # 5. 更新索引
            self.current_vnf_index = idx + 1

            print(f"✅ VNF{vnf_type} 部署成功 @ 节点{node_id}")
            return True

        except Exception as e:
            print(f"❌ 部署异常: {e}")
            return False

    def _validate_sfc_integrity(self):
        """
        [SFC 完整性验证]
        确保从 Source -> VNF1 -> ... -> VNFn -> 每个Dest 的路径在 Tree 中是连通的。
        """
        try:
            # 这是一个简化版的验证，因为我们是 step-by-step 记录的 tree，
            # 只要步骤是连续的，通常路径就是连通的。
            # 这里我们只检查数量是否对得上即可。

            dests = self.current_request.get('dest', [])
            connected = self.current_tree.get('connected_dests', set())

            if len(connected) < len(dests):
                return False

            return True
        except:
            return False

    def _calculate_cycle_penalty(self, target_node):
        """
        🔥 更智能的循环检测和惩罚
        检测模式而不是简单禁止
        """
        if not hasattr(self, '_visited_history'):
            self._visited_history = []

        current_node = self.current_node_location

        # 记录当前步 (临时记录用于计算，实际更新在 _update_movement_history)
        # 注意：这里我们只是计算针对 target_node 的潜在惩罚，不更新历史

        # 如果历史太短，不惩罚
        if len(self._visited_history) < 2:
            return 0.0

        penalty = 0.0

        # 1. 检测立即回头：A->B->A
        last_move = self._visited_history[-1]  # (from, to)
        # 如果上一步是从 target_node 走过来的，现在又要回去
        if last_move[1] == current_node and last_move[0] == target_node:
            penalty -= 1.5  # 立即回头：中等惩罚

        # 2. 检测短循环：A->B->C->B
        # 取最近访问过的节点列表
        recent_nodes = [move[1] for move in self._visited_history[-6:]]
        if target_node in recent_nodes:
            freq = recent_nodes.count(target_node)
            if freq > 1:
                penalty -= 0.5 * freq

        # 3. 检测振荡模式：A->B->A->B
        if len(self._visited_history) >= 3:
            moves = self._visited_history[-3:]
            # 检查是否形成 B->A, A->B, B->A 的趋势
            if (moves[0][0] == target_node and moves[0][1] == current_node and  # 上上步是从 target 来的
                    moves[1][0] == current_node and moves[1][1] == target_node and  # 上一步去了 target
                    moves[2][0] == target_node and moves[2][1] == current_node):  # 这一步又回到了 current
                penalty -= 3.0  # 振荡模式：重罚

        return penalty

    def _update_movement_history(self, from_node, to_node):
        """更新移动历史"""
        if not hasattr(self, '_visited_history'):
            self._visited_history = []

        # 记录移动
        self._visited_history.append((from_node, to_node))

        # 保持历史长度
        if len(self._visited_history) > 20:
            self._visited_history.pop(0)

    def _is_node_visited_too_often(self, node):
        """检查节点是否被访问过于频繁"""
        if not hasattr(self, '_visited_history'):
            return False

        # 统计最近10步中该节点出现的次数
        recent_steps = self._visited_history[-10:] if len(self._visited_history) >= 10 else self._visited_history
        count = recent_steps.count(node)

        # 如果最近10步中出现了超过3次，则认为过于频繁
        return count > 3
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

    def find_deployable_nodes(self):
        """查找所有可部署的节点（用于调试）"""
        if not self.current_request: return []

        vnf_list = self.current_request.get('vnf', [])
        if self.current_vnf_index >= len(vnf_list): return []

        deployable = []
        # 简单检查逻辑
        for node in range(self.n):
            if self._check_deployment_validity(node):
                # 假设资源足够
                deployable.append(node)
        return deployable

    def _get_distance(self, u, v):
        """[辅助方法] 计算距离，防止报错"""
        if u == v: return 0
        try:
            # 优先用 TopologyMgr
            if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'get_distance'):
                return self.topology_mgr.get_distance(u, v)
            # 备用 NetworkX
            import networkx as nx
            if not hasattr(self, '_nx_graph'):
                if hasattr(self, 'topology_matrix'):
                    self._nx_graph = nx.from_numpy_array(self.topology_matrix)
                else:
                    return 50  # 无法计算时给个默认值
            return nx.shortest_path_length(self._nx_graph, u, v)
        except:
            return 50  # 出错兜底

    def _init_path_planner(self):
        """初始化路径规划缓存"""
        self._path_cache = {}

    def _find_best_path_to_unconnected(self, start_node, unconnected_dests):
        """找到去往任意一个未连接Dest的最短路径"""
        best_path = None
        min_len = 9999

        for dest in unconnected_dests:
            # 优先查缓存
            cache_key = (start_node, dest)
            if hasattr(self, '_path_cache') and cache_key in self._path_cache:
                path = self._path_cache[cache_key]
            else:
                path = self._a_star_search(start_node, dest)
                if hasattr(self, '_path_cache'):
                    self._path_cache[cache_key] = path

            if path and len(path) < min_len:
                min_len = len(path)
                best_path = path

        return best_path

    def _a_star_search(self, start, goal):
        """标准的 A* 搜索算法 - 修复版"""
        if start == goal:
            return [start]

        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        def heuristic(n):
            return 0

        f_score = {start: heuristic(start)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # 🔥 修复：直接遍历所有节点，检查是否有链接
            neighbors = []
            for v in range(self.n):
                if v != current and self.resource_mgr.has_link(current, v):
                    neighbors.append(v)

            for neighbor in neighbors:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _find_path(self, start, end):
        """BFS 寻路 (用于 Mask 计算)"""
        return self._a_star_search(start, end)  # 复用A*逻辑


    def _get_vnf_hub_nodes(self):
        """获取所有VNF部署节点"""
        hub_nodes = set()
        if self.current_tree.get('placement', {}):
            for key, node in self.current_tree['placement'].items():
                hub_nodes.add(node)
        return hub_nodes


    def print_connection_status(self):
        """打印连接状态"""
        if not self.current_request or self.current_vnf_index < len(self.current_request.get('vnf', [])):
            return

        dests = self.current_request.get('dest', [])
        if 'connected_dests' not in self.current_tree:
            return

        connected = self.current_tree['connected_dests']
        unconnected = [d for d in dests if d not in connected]

        print(f"\n📊 连接状态: {len(connected)}/{len(dests)}")
        if unconnected:
            print(f"   未连接: {unconnected}")
            print(f"   当前位置: {self.current_node_location}")

            # 计算到每个未连接节点的距离
            distances = []
            for dest in unconnected:
                path = self._find_path(self.current_node_location, dest)
                if path:
                    distances.append((dest, len(path) - 1))
                else:
                    distances.append((dest, 999))

            # 按距离排序
            distances.sort(key=lambda x: x[1])
            print(f"   距离排序:")
            for dest, dist in distances[:3]:  # 显示最近的3个
                if dist < 999:
                    print(f"      {dest}: {dist}跳")
                else:
                    print(f"      {dest}: 不可达")


    def _find_nearest_unconnected(self, start_node, unconnected_dests):
        """找到最近的未连接目的节点"""
        nearest = None
        min_dist = float('inf')
        path_to_nearest = None

        for dest in unconnected_dests:
            path = self._find_path(start_node, dest)
            if path:
                dist = len(path) - 1
                if dist < min_dist:
                    min_dist = dist
                    nearest = dest
                    path_to_nearest = path

        if nearest:
            return (nearest, min_dist, path_to_nearest)
        return None


    def print_navigation_guide(self):
        """打印导航指南"""
        if not self.current_request:
            return

        req = self.current_request
        vnf_list = req.get('vnf', [])
        dests = req.get('dest', [])

        if self.current_vnf_index < len(vnf_list):
            # 部署阶段
            print(f"\n💡 [部署阶段] 需要部署 {len(vnf_list)} 个VNF，已部署 {self.current_vnf_index} 个")
            print(f"   当前节点: {self.current_node_location}")
            print(f"   DC节点: {self.dc_nodes}")

            # 找出可部署的DC节点
            deployable = []
            for dc in self.dc_nodes:
                if dc != req.get('source') and dc not in dests:
                    if self._check_deployment_validity(dc):
                        deployable.append(dc)

            if deployable:
                print(f"   可部署的DC节点: {deployable}")
            else:
                print(f"   ⚠️ 没有可部署的DC节点！检查资源或拓扑")

        else:
            # 树构建阶段
            if 'connected_dests' not in self.current_tree:
                return

            connected = self.current_tree['connected_dests']
            unconnected = [d for d in dests if d not in connected]

            if unconnected:
                print(f"\n🗺️ [导航指南] 已连接 {len(connected)}/{len(dests)}，剩余 {len(unconnected)} 个")
                print(f"   当前位置: {self.current_node_location}")
                print(f"   未连接节点: {unconnected}")

                # 距离排序
                distances = []
                for dest in unconnected:
                    path = self._find_path(self.current_node_location, dest)
                    if path:
                        distances.append((dest, len(path) - 1, path))

                if distances:
                    distances.sort(key=lambda x: x[1])
                    print(f"   距离排序:")
                    for i, (dest, dist, path) in enumerate(distances[:3]):  # 显示最近的3个
                        print(f"     {i + 1}. 节点{dest}: {dist}跳 - 路径: {path}")

    def _diagnose_connectivity_failure(self, step_idx):
        """
        🚑 [深度诊断 - 修复版] 诊断连接失败原因
        修复了 get_link_bandwidth 报错，增加了直接读取资源字典的兼容性
        """
        print(f"\n🔍 [DCC诊断] Step {step_idx} | 当前节点: {self.current_node_location}")

        # 1. 识别剩余目标
        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())
        unconnected = [d for d in dests if d not in connected]

        print(f"   📉 未连接目标: {unconnected}")

        if not unconnected:
            print("   ✅ 所有目标已连接 (无需诊断)")
            return

        # 2. 获取当前 Mask 和 邻居
        mask = self.get_low_level_action_mask()
        if hasattr(self, 'resource_mgr'):
            neighbors = self.resource_mgr.get_neighbors(self.current_node_location)
        else:
            neighbors = self.topology_mgr.get_neighbors(self.current_node_location)

        print(f"   🏠 物理邻居: {neighbors}")
        print(f"   🎭 当前Mask允许: {[n for n in neighbors if mask[n]]}")

        # 3. 逐个分析未连接节点
        for dest in unconnected:
            print(f"   🎯 分析目标 Node {dest}:")

            # --- A. 物理路径检查 (A*) ---
            path = self._find_path(self.current_node_location, dest)
            if not path:
                print(f"      ❌ [物理层] 致命：物理拓扑不连通！无法到达。")
                continue

            # 获取下一跳
            next_hop = path[1] if len(path) > 1 else path[0]
            print(f"      ✅ [物理层] 最短路径: {path} (下一跳: {next_hop})")

            # --- B. Mask 阻断检查 ---
            if not mask[next_hop]:
                print(f"      ❌ [逻辑层] Mask 封锁了最佳下一跳 {next_hop}！")

                # 深入分析 Mask 为什么封锁
                visit_count = 0
                if hasattr(self, 'node_visit_counts'):
                    visit_count = self.node_visit_counts.get(next_hop, 0)

                print(f"         - 访问频次: {visit_count}")

                if visit_count >= 3:
                    print(f"         - 原因: 访问次数过多，触发防死循环锁死。")
                else:
                    print(f"         - 原因: 可达性检测认为那是死胡同，或者是黑名单节点。")
            else:
                print(f"      ✅ [逻辑层] Mask 允许通过。")

            # --- C. 资源/带宽检查 (🔥 核心修复部分) ---
            # 尝试多种方式获取带宽，防止报错
            bw = None
            link = (self.current_node_location, next_hop)

            # 方式1: 尝试调用方法
            if hasattr(self.resource_mgr, 'get_link_bandwidth'):
                try:
                    bw = self.resource_mgr.get_link_bandwidth(self.current_node_location, next_hop)
                except:
                    pass

            # 方式2: 直接访问 links 字典 (这是通常的 SDN 环境结构)
            if bw is None and hasattr(self.resource_mgr, 'links'):
                if isinstance(self.resource_mgr.links, dict):
                    # links 可能包含 'bandwidth' 键
                    if 'bandwidth' in self.resource_mgr.links:
                        bw = self.resource_mgr.links['bandwidth'].get(link)
                        if bw is None:  # 尝试反向链路
                            bw = self.resource_mgr.links['bandwidth'].get((next_hop, self.current_node_location))

            # 方式3: 访问拓扑矩阵 (如果 links 字典不可用)
            if bw is None and hasattr(self.resource_mgr, 'topology'):
                try:
                    bw = self.resource_mgr.topology[self.current_node_location][next_hop]
                except:
                    pass

            # 打印结果
            if bw is not None:
                print(f"      💰 [资源层] 链路 {link} 带宽: {bw}")
                if bw <= 0:
                    print(f"         ❌ 带宽耗尽！这可能是 Agent 不走这条路的原因。")
            else:
                print(f"      ⚠️ [资源层] 无法读取链路带宽信息 (属性缺失)")

        print("=" * 50)

    def _is_good_hub(self, node_id, dests):
        """判断是否是优质枢纽 (平均跳数 < 5)"""
        total_dist = 0
        valid = 0
        for d in dests:
            path = self._find_path(node_id, d)
            if path:
                total_dist += (len(path) - 1)
                valid += 1
        return (total_dist / valid) <= 5.0 if valid > 0 else False

    def _find_nearest_valid_dc(self, current_node):
        """找到最近的合规 DC"""
        best_dc = -1
        min_dist = 999

        for dc in self.dc_nodes:
            if self._check_deployment_validity(dc):
                path = self._find_path(current_node, dc)
                if path:
                    dist = len(path) - 1
                    if dist < min_dist:
                        min_dist = dist
                        best_dc = dc
        return best_dc

    def _check_link_validity(self, from_node, to_node):
        """检查链路有效性"""
        try:
            if hasattr(self, 'resource_mgr'):
                return self.resource_mgr.has_link(from_node, to_node)
            else:
                return (self.topo[from_node, to_node] > 0)
        except:
            return True

    def _min_distance_to_unconnected(self, node, unconnected):
        """计算到未连接节点的最小距离"""
        if not unconnected:
            return 0

        min_dist = 999
        for dest in unconnected:
            try:
                path = self.topology_mgr.get_shortest_path(node, dest)
                if path:
                    dist = len(path) - 1
                    min_dist = min(min_dist, dist)
            except:
                pass

        return min_dist

    def _diagnose_illegal_action(self, current_node, target_node, vnf_list, dests):
        """诊断非法动作（保留你原来的诊断日志）"""
        print(f"\n{'=' * 60}")
        print(f"❌ [动作被禁止诊断]")
        print(f"   当前位置: {current_node}")
        print(f"   目标位置: {target_node}")

        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        if is_vnf_complete:
            print(f"   阶段: 树构建")

            connected = self.current_tree.get('connected_dests', set())
            unconnected = [d for d in dests if d not in connected]

            print(f"   已连接: {list(connected)} ({len(connected)}/{len(dests)})")
            print(f"   未连接: {unconnected}")
            print(f"   目标节点是未连接的目的? {target_node in unconnected}")

            # 物理连接性
            try:
                neighbors = self.resource_mgr.get_neighbors(current_node) if hasattr(self, 'resource_mgr') else []
                print(f"   当前位置的物理邻居: {neighbors}")
                print(f"   目标节点是邻居? {target_node in neighbors}")

                path = self.topology_mgr.get_shortest_path(current_node, target_node)
                if path:
                    print(f"   最短路径: {path} (长度={len(path) - 1})")
                else:
                    print(f"   ❌ 无路径到目标节点！")
            except Exception as e:
                print(f"   路径查找错误: {e}")

            # visit_count
            if hasattr(self, 'node_visit_counts'):
                vc = self.node_visit_counts.get(target_node, 0)
                print(f"   visit_count[目标{target_node}] = {vc}")
        else:
            print(f"   阶段: VNF部署")
            print(f"   已部署: {deployed_count}/{len(vnf_list)}")

        mask = self.get_low_level_action_mask()
        valid_actions = np.where(mask)[0]
        print(f"   可用动作({len(valid_actions)}个): {valid_actions.tolist()}")
        print(f"{'=' * 60}\n")

    def _update_tree_state(self, u, v):
        """简单更新树拓扑和位置"""
        if 'tree' not in self.current_tree:
            self.current_tree['tree'] = {}

        # 记录边 (仅用于拓扑记录，不涉及资源，资源在Step里扣了)
        # 这里用 1.0 或 bw 都可以，通常用于GNN输入
        self.current_tree['tree'][(u, v)] = 1.0

        self.nodes_on_tree.add(u)
        self.nodes_on_tree.add(v)

    # ============================================
    # 3. 新增 get_action_mask 方法
    # ============================================
    def get_action_mask(self) -> np.ndarray:
        """
        获取动作掩码（用于RL Agent）

        Returns:
            np.ndarray: [n_actions] 掩码
                1.0 = 完全有效
                0.5 = 勉强可用（冷却即将结束）
                0.0 = 无效（被黑名单拉黑）
        """
        import numpy as np

        # 获取有效动作
        valid_actions = self.get_valid_actions()

        # 创建掩码
        mask = np.zeros(self._n_actions, dtype=np.float32)

        # 特殊情况：所有动作无效
        if not valid_actions or valid_actions[0] == -1:
            logger.error("❌ 所有动作无效")
            return mask

        # 标记有效动作
        for action in valid_actions:
            if 0 <= action < self._n_actions:
                mask[action] = 1.0

        # ✅ 应急：添加"勉强可用"的动作
        if hasattr(self, 'blacklist_manager'):
            if mask.sum() < self.blacklist_manager._min_valid_actions:
                logger.warning(f"⚠️ 可用动作仅{mask.sum()}个，添加勉强可用动作")

                blacklisted = self.blacklist_manager.get_blacklisted_nodes()
                for node_id in blacklisted:
                    if 0 <= node_id < self._n_actions:
                        info = self.blacklist_manager._blacklist.get(node_id)
                        if info:
                            remaining = max(0, info['cooldown_until'] - self.blacklist_manager._step_counter)

                            # 剩余冷却 <= 2步，标记为"勉强可用"
                            if remaining <= 2:
                                if self._check_node_resources(node_id):
                                    mask[node_id] = 0.5
                                    logger.debug(f"🔄 节点{node_id}冷却即将结束，标记为勉强可用")

        return mask

    # ============================================
    # 4. 新增 get_valid_actions 方法
    # ============================================
    def get_valid_actions(self, state=None):
        """
        获取有效动作（排除黑名单节点）

        Returns:
            List[int]: 有效动作列表
        """
        if not hasattr(self, 'blacklist_manager'):
            # 如果没有黑名单管理器，使用原有逻辑
            return self._get_base_valid_actions()

        # 1. ✅ 定期清理黑名单
        if self.blacklist_manager.should_clean():
            self.blacklist_manager.clean_expired(
                resource_checker=self._check_node_resources
            )
            self.blacklist_manager.mark_cleaned()

        # 2. 获取基础有效动作
        valid_actions = self._get_base_valid_actions()

        # 3. ✅ 排除黑名单节点
        blacklisted = self.blacklist_manager.get_blacklisted_nodes()
        valid_actions = [a for a in valid_actions if a not in blacklisted]

        # 4. ✅ 应急清理（可用动作太少）
        if len(valid_actions) < self.blacklist_manager._min_valid_actions and len(blacklisted) > 0:
            logger.warning(
                f"⚠️ 可用动作仅{len(valid_actions)}个，"
                f"黑名单{len(blacklisted)}个，触发应急清理"
            )
            self.blacklist_manager.emergency_clean(
                resource_checker=self._check_node_resources
            )

            # 重新获取
            valid_actions = self._get_base_valid_actions()
            blacklisted = self.blacklist_manager.get_blacklisted_nodes()
            valid_actions = [a for a in valid_actions if a not in blacklisted]

        # 5. 兜底
        if len(valid_actions) == 0:
            logger.error("❌ 所有节点资源不足或在冷却中，任务失败")
            valid_actions = [-1]

        return valid_actions

    # ============================================
    # 5. 新增 _get_base_valid_actions 方法
    # ============================================
    def _get_base_valid_actions(self):
        """
        获取基础有效动作（不考虑黑名单）

        这是原有的 get_valid_actions 逻辑
        """
        # 如果没有当前请求，返回所有节点
        if self.current_request is None:
            return list(range(self.n))

        # 获取当前位置的邻居节点
        current_node = self.current_node_location

        # 获取邻居
        neighbors = []
        try:
            if hasattr(self, 'resource_mgr'):
                neighbors = self.resource_mgr.get_neighbors(current_node)
            else:
                neighbors = np.where(self.topo[current_node] > 0)[0].tolist()
        except:
            neighbors = []

        # 判断是否在VNF部署阶段
        vnf_list = self.current_request.get('vnf', [])
        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        valid_actions = []

        if not is_vnf_complete:
            # VNF阶段：邻居节点 + 当前节点（如果可部署）
            valid_actions.extend(neighbors)

            # 检查当前节点是否可以部署
            if current_node in getattr(self, 'dc_nodes', []):
                if self._check_deployment_validity(current_node):
                    valid_actions.append(current_node)
        else:
            # 树构建阶段：邻居节点 + 当前节点（如果是未连接目的地）
            valid_actions.extend(neighbors)

            # 检查当前节点是否是未连接目的地
            connected = self.current_tree.get('connected_dests', set())
            dests = self.current_request.get('dest', [])
            if current_node in dests and current_node not in connected:
                valid_actions.append(current_node)

        # 去重
        valid_actions = list(set(valid_actions))

        return valid_actions if valid_actions else [0]

    # ============================================
    # 6. 新增 _check_node_resources 方法
    # ============================================
    def _check_node_resources(self, node_id: int) -> bool:
        """
        检查节点资源是否充足

        Args:
            node_id: 节点ID

        Returns:
            True: 资源充足
            False: 资源不足
        """
        try:
            if self.current_request is None:
                return True

            # 获取当前VNF需求
            vnf_list = self.current_request.get('vnf', [])
            if not vnf_list:
                return True

            # 获取当前需要部署的VNF索引
            deployed_count = len(self.current_tree.get('placement', {}))
            if deployed_count >= len(vnf_list):
                return True  # 所有VNF已部署

            vnf_idx = deployed_count

            # 获取资源需求
            vnf_cpu = self.current_request.get('vnf_cpu', [1.0] * len(vnf_list))
            required_cpu = vnf_cpu[vnf_idx] if vnf_idx < len(vnf_cpu) else 1.0

            # 检查节点资源
            if hasattr(self, 'resource_mgr'):
                available_cpu = self.resource_mgr.get_node_cpu(node_id)
            else:
                # Fallback
                available_cpu = 100.0

            # 留10%余量
            return available_cpu >= required_cpu * 1.1

        except Exception as e:
            logger.error(f"检查节点{node_id}资源时出错: {e}")
            return False