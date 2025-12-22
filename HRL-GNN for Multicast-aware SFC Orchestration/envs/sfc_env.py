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
        # 11. 初始化奖励评估器 [关键修改]
        # ========================================
        training_phase = 3

        # 从 config 中获取 reward 参数，默认为空字典
        reward_params = config.get('reward', {})

        # 🔥 修改点：移除了 epoch 和 max_epochs 参数
        # 新版 RewardCritic 只接收 training_phase 和 params
        self.reward_critic = RewardCritic(
            training_phase=training_phase,
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
    def reset(self, *, seed=None, options=None):
        """Gym 标准 Reset (强制随机化修复版)"""
        super().reset(seed=seed)

        phase = "phase3"
        if options is not None:
            phase = options.get("phase", phase)

        self._reset_core()

        # --- 1. 数据加载逻辑 ---
        if not hasattr(self.data_loader, 'requests') or len(self.data_loader.requests) == 0:
            if not self.load_dataset(phase):
                logger.warning(f"⚠️ Failed to load dataset for {phase}")

        # =====================================================
        # 🔥【强制随机化】每次 Reset 都打乱请求顺序
        # =====================================================
        if hasattr(self.data_loader, 'requests') and len(self.data_loader.requests) > 0:
            import random

            # 完全随机打乱
            random.shuffle(self.data_loader.requests)

            # 🔥 重置请求索引（关键！）
            self._request_index = 0

            # 调试日志
            first_req_id = self.data_loader.requests[0].get('id', 'Unknown')
            first_req_src = self.data_loader.requests[0].get('source', 'Unknown')
            print(f"🔄 [RESET] 数据集已打乱，首个请求 ID={first_req_id}, Source={first_req_src}")

        # --- 3. 获取第一个请求 ---
        req, _ = self.reset_request()

        if req is None:
            return self.get_state(), {"phase": phase, "error": "no_requests"}

        obs = self.get_state()
        info = {"phase": phase}
        return obs, info

    def reset_request(self) -> Tuple[Optional[Dict], Any]:
        """重置请求（全自动修复 1-based 数据偏移）"""
        self.policy_helper.clear_cache()
        self.current_request = None
        self._prev_dist = None
        self.reward_critic.on_new_request()

        if not hasattr(self, '_request_index'):
            self._request_index = 0

        if self._request_index >= len(self.data_loader.requests):
            logger.info("[Env] No more requests")
            return None, self.get_state()

        # 读取原始请求
        raw_req = self.data_loader.requests[self._request_index]
        self._request_index += 1
        self.total_requests_seen += 1

        # 🔥🔥🔥【数据清洗】🔥🔥🔥
        req = raw_req.copy()

        # 1. 修复 Node ID (如果发现 >= 28，说明是 1-based)
        src = req.get("source", 0)
        if src >= self.n:
            src -= 1
            req['source'] = src

        new_dests = []
        for d in req.get("dest", []):
            new_dests.append(d - 1 if d >= self.n else d)
        req['dest'] = new_dests

        # 2. 修复 VNF Type ID (如果发现 >= 8，说明是 1-based)
        # 您的 K_vnf 是 8，所以合法索引是 0~7
        new_vnfs = []
        for v in req.get('vnf', []):
            if v >= self.K_vnf:
                new_vnfs.append(v - 1)
            else:
                new_vnfs.append(v)
        req['vnf'] = new_vnfs

        # 3. 初始化状态
        self.current_request = req
        self.unadded_dest_indices = set(range(len(new_dests)))

        # 确保位置初始化正确
        start_node = req['source']
        self.current_node_location = start_node
        self.nodes_on_tree = {start_node}

        # 重置 Tree
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {}
        }

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
    def _get_action_metrics(self, node_action: int, tree_edges: Dict) -> Tuple[float, float, int]:
        """
        辅助方法：从资源管理器中提取当前动作的物理指标
        返回: (cpu_remain, min_bandwidth, total_hops)
        """
        # 1. 获取选中节点的剩余 CPU
        try:
            # 根据 resource.py，self.C 是存储节点 CPU 的 numpy 数组
            cpu_remain = float(self.resource_mgr.C[node_action])
        except (IndexError, AttributeError) as e:
            logger.warning(f"无法获取节点 {node_action} 的 CPU: {e}")
            cpu_remain = 0.0

        # 2. 获取本次部署涉及链路的瓶颈带宽
        min_bw = 99999.0
        hops = 0

        if tree_edges:
            hops = len(tree_edges)
            for edge_key, _ in tree_edges.items():
                # 解析 edge_key, 可能是 tuple (u, v) 或 string "u-v"
                u, v = None, None
                if isinstance(edge_key, tuple):
                    u, v = edge_key
                elif isinstance(edge_key, str):
                    try:
                        u, v = map(int, edge_key.strip('()').split('-'))
                    except ValueError:
                        continue

                if u is not None and v is not None:
                    # 从 resource_mgr 中获取链路带宽
                    # 优先使用 links['bandwidth'] 字典 (最准确，包含了动态扣除后的值)
                    if (u, v) in self.resource_mgr.links['bandwidth']:
                        bw = self.resource_mgr.links['bandwidth'][(u, v)]
                        if bw < min_bw:
                            min_bw = float(bw)
                    # 备选：尝试从 edge_to_phys 映射去查 self.B 数组
                    elif hasattr(self.resource_mgr, 'edge_to_phys') and (u, v) in self.resource_mgr.edge_to_phys:
                        phys_id = self.resource_mgr.edge_to_phys[(u, v)]
                        if phys_id < len(self.resource_mgr.B):
                            bw = self.resource_mgr.B[phys_id]
                            if bw < min_bw:
                                min_bw = float(bw)

            # 如果循环完了 min_bw 还是初始值，说明没找到有效链路信息
            if min_bw > 90000.0:
                min_bw = 0.0
        else:
            # 如果没有边 (例如源节点=目的节点)，带宽设为 0
            min_bw = 0.0

        return cpu_remain, min_bw, hops



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

    def step_low_level(self, action):
        """
        [最终修复版] 分离移动和部署逻辑
        """
        self.step_counter += 1

        # 基础验证
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        if not (0 <= action < self.n):
            return self.get_state(), -10.0, True, False, {'error': 'invalid_action'}
        if self.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_request'}

        current_node = getattr(self, 'current_node_location',
                               self.current_request.get('source', 0))
        target_node = int(action)

        # 判断意图：是想部署还是想移动？
        # 如果当前位置符合部署条件，系统期望 Agent 选当前节点来触发部署
        should_deploy = self._should_deploy_at_current_node()

        if should_deploy:
            # --- 部署模式 ---
            if target_node != current_node:
                # 惩罚：明明该部署了（到了金矿），却跑了
                info = {
                    'error': 'deploy_mode_mismatch',
                    'msg': f'Expect deploy at {current_node}, got move to {target_node}'
                }
                # 这是一个严重错误，给予重罚但允许继续（或直接判负）
                return self.get_state(), -5.0, False, False, info

            # 执行部署
            return self._execute_deployment(current_node)

        else:
            # --- 移动模式 ---
            # 无论 target_node 是当前节点（等待）还是邻居（移动），都走移动逻辑
            return self._execute_movement(current_node, target_node)

    def get_low_level_action_mask(self) -> np.ndarray:
        """
        [最终修正版] 动态 Mask
        1. 部署模式 -> 只能选当前节点 (Action = Current Node)
        2. 移动模式 -> 只能选邻居节点 (Action ∈ Neighbors, Exclude Current)
        """
        mask = np.zeros(self.NB_LOW_LEVEL_ACTIONS, dtype=np.bool_)

        if self.current_request is None:
            return np.ones(self.NB_LOW_LEVEL_ACTIONS, dtype=np.bool_)

        current_node = getattr(self, 'current_node_location',
                               self.current_request.get('source', 0))

        # --- A. 部署模式 ---
        # 如果当前位置满足部署条件，强制 Agent 选择原地不动来触发部署
        if self._should_deploy_at_current_node():
            if 0 <= current_node < self.n:
                mask[current_node] = True
            else:
                mask[:] = True
            return mask

        # --- B. 移动模式 ---
        # 如果不需要部署，强制 Agent 移动到邻居 (禁止原地发呆！)
        if hasattr(self, 'topology_mgr'):
            neighbors = self.topology_mgr.get_neighbors(current_node)
        else:
            neighbors = []

        valid_count = 0
        for node in neighbors:
            if 0 <= node < self.n:
                mask[node] = True
                valid_count += 1

        # 兜底：如果真的被围死无路可走，才允许原地等待
        if valid_count == 0:
            if 0 <= current_node < self.n:
                mask[current_node] = True
            else:
                mask[:] = True

        return mask
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