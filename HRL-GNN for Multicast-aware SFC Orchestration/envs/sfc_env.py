# modules/core_env.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC_HIRL_Env - 完整可运行的主环境类（分层强化学习 + 多播感知）
已完全模块化，职责清晰，兼容 Flat 和 GNN 两种状态表示
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import gym
from gym import spaces

from envs.modules.resource import ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.path_manager import PathManager
from envs.modules.event_handler import EventHandler
from envs.modules.utils.metrics import MetricsTracker
from envs.modules.policy_helper import PolicyHelper

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
    - 可切换 Flat / GNN 状态表示（通过子类 SFC_HIRL_Env_GNN）
    """

    def __init__(self, config: Dict, use_gnn: bool = False):
        super().__init__()

        self.cfg = config
        self.use_gnn = use_gnn

        # ============================ 基础路径与配置 ============================
        self.input_dir = Path(config['path']['input_dir'])
        self.failure_output_dir = Path(config['path'].get('failure_output_dir',
                                    self.input_dir.parent / "out_failure"))
        os.makedirs(self.failure_output_dir, exist_ok=True)

        self.enable_render = config.get('render', {}).get('enable_failure_visualization', False)

        # ============================ 核心模块初始化 ============================
        topo = config['topology']['matrix']                    # np.ndarray (N, N)
        dc_nodes = config['topology']['dc_nodes']              # List[int]
        capacities = config['capacities']                      # Dict

        self.resource_mgr = ResourceManager(topo, capacities, dc_nodes)
        self.data_loader = DataLoader(config)
        self.path_manager = PathManager(max_paths=config.get('env', {}).get('max_cached_paths', 10))
        self.event_handler = EventHandler(self.resource_mgr)
        self.metrics = MetricsTracker()

        # 策略辅助（专家 + 备份）
        self.policy_helper = PolicyHelper(self.input_dir, topo, dc_nodes, capacities)

        # 可视化与日志
        if self.enable_render:
            self.visualizer = FailureVisualizer(topo=topo)
        self.vnf_logger = VNFMetricsLogger()

        # 奖励批评家
        self.reward_critic = RewardCritic()

        # ============================ 环境参数 ============================
        self.num_nodes = self.resource_mgr.num_nodes
        self.K_vnf = self.resource_mgr.K_vnf
        self.NB_HIGH_LEVEL_GOALS = config.get('env', {}).get('nb_high_level_goals', 10)
        self.NB_LOW_LEVEL_ACTIONS = config.get('env', {}).get('nb_low_level_actions', 50)

        # ============================ 动作与观察空间 ============================
        # 高层动作：选择哪个目标分支（dest_idx）
        self.high_level_action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        # 低层动作：i_idx * K_path + k_idx
        self.low_level_action_space = spaces.Discrete(self.NB_LOW_LEVEL_ACTIONS)

        # 观察空间占位（实际会在子类中覆盖为具体 Dict 或 Box）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # ============================ 运行时状态 ============================
        self.current_request: Optional[Dict] = None
        self.current_tree: Dict[str, Any] = {'tree': np.zeros(self.resource_mgr.L),
                                             'hvt': np.zeros((self.num_nodes, self.K_vnf)),
                                             'paths_map': {}}
        self.unadded_dests: List[int] = []          # 尚未接入的多播目标
        self.nodes_on_tree: set = set()            # 当前树上节点集合
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.phase_done = False

        # 状态访问计数（用于探索奖励）
        self.state_visit_counter = {}

    # =====================================================================
    # 数据加载与重置
    # =====================================================================
    def load_dataset(self, phase: str = "phase3") -> bool:
        return self.data_loader.load_dataset(phase)

    def reset(self, phase: str = "phase3"):
        self.resource_mgr.reset()
        self.data_loader.reset()
        self.path_manager.reset()
        self.event_handler.reset()
        self.metrics.reset()
        self.policy_helper.clear_cache()

        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.state_visit_counter.clear()

        if not self.load_dataset(phase):
            raise RuntimeError(f"Failed to load dataset for {phase}")

        self._reset_current_request()
        return self.get_state()

    def _reset_current_request(self):
        """获取下一个到达的请求并初始化树结构"""
        arrivals = self.data_loader.get_current_arrivals()
        self.current_request = arrivals[0] if arrivals else None

        if self.current_request is None:
            self.phase_done = True
            return

        self.total_requests_seen += 1
        req = self.current_request
        self.unadded_dests = req.get('dest', []).copy()
        self.nodes_on_tree = {req['source'] - 1}  # 0-based

        # 重置当前树
        self.current_tree = {
            'tree': np.zeros(self.resource_mgr.L, dtype=np.float32),
            'hvt': np.zeros((self.num_nodes, self.K_vnf), dtype=np.float32),
            'paths_map': {}
        }
        self.path_manager.reset()
        self.policy_helper.clear_cache()

    # =====================================================================
    # 状态获取接口（Flat / GNN 统一入口）
    # =====================================================================
    def _get_state(self):
        """统一状态获取入口"""
        if self.use_gnn:
            # 调用 GNN 状态提取
            return self.resource_mgr.get_graph_state(
                current_request=self.current_request,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree,
                served_dest_count=len(self.served_destinations),
                sharing_strategy=self.sharing_strategy,
                nb_high_goals=self.NB_HIGH_LEVEL_GOALS
            )
        else:
            # 调用 Flat 状态提取
            return self.resource_mgr.get_flat_state(
                current_request=self.current_request,
                unadded_dest_indices=self.unadded_dest_indices,
                nodes_on_tree=self.nodes_on_tree,
                current_tree=self.current_tree
            )
    # =====================================================================
    # 分层 Step 接口
    # =====================================================================
    def step_high_level(self, goal_idx: int):
        """
        高层动作：选择下一个要接入的目标（dest_idx）
        返回：state, reward, done, info
        """
        if goal_idx >= len(self.unadded_dests):
            return self.get_state(), -1.0, False, {"invalid_action": True}

        # 选择目标，进入低层决策阶段
        self.current_goal_idx = goal_idx
        self.current_dest = self.unadded_dests[goal_idx]

        info = {
            "high_level_goal": self.current_dest,
            "remaining_dests": len(self.unadded_dests)
        }
        return self.get_state(), 0.0, False, info

    def step_low_level(self, action: int):
        """
        低层动作：选择路径索引 i 和 k_path
        返回：state, reward, sub_done（当前目标完成）, req_done（整个请求完成）, info
        """
        if self.current_request is None:
            return self.get_state(), 0.0, True, True, {"done": True}

        # 解析动作
        k_idx = action % self.policy_helper.K_path
        i_idx = action // self.policy_helper.K_path

        network_state = self.resource_mgr.get_network_state_dict(self.current_request)

        feasible, plan, backup_used, reason = self.policy_helper.get_best_plan(
            request=self.current_request,
            network_state=network_state,
            goal_dest_idx=self.current_goal_idx,
            k_idx=k_idx,
            i_idx=i_idx,
            current_tree=self.current_tree,
            nodes_on_tree=self.nodes_on_tree,
            path_manager=self.path_manager
        )

        reward = 0.0
        sub_done = False
        info = {"backup_used": backup_used, "feasible": feasible}

        if feasible and plan is not None:
            # 成功部署
            success = self.resource_mgr.apply_deployment(plan, self.current_request)
            if success:
                # 更新树结构
                self.current_tree['tree'] += plan['tree']
                self.current_tree['hvt'] += plan['hvt']
                self.current_tree['paths_map'][self.current_dest] = plan['new_path_full']

                # 更新已接入节点
                new_nodes = set(np.where(plan['hvt'] > 0)[0])
                self.nodes_on_tree.update(new_nodes)

                # 移除已接入目标
                self.unadded_dests.pop(self.current_goal_idx)

                # VNF 共享记录
                for node, vnf_t in np.argwhere(plan['hvt'] > 0):
                    self.resource_mgr.share_vnf(node + 1, vnf_t, self.current_goal_idx)

                reward = self.reward_critic.compute_reward(
                    success=True, backup_used=backup_used,
                    sharing_rate=self.resource_mgr.get_vnf_sharing_rate()
                )
                sub_done = True
                info["success"] = True
            else:
                reward = -0.5
                info["reason"] = "resource_insufficient_after_expert"
        else:
            # 失败：可视化 + 惩罚
            if self.enable_render:
                self.render_failure(self.current_goal_idx, title=f"Fail_Req{self.current_request['id']}")
            reward = self.reward_critic.compute_reward(success=False, backup_used=backup_used)
            info["reason"] = reason or "expert_failed"

        # 检查当前请求是否全部完成
        req_done = len(self.unadded_dests) == 0
        if req_done:
            self.total_requests_accepted += 1
            self.metrics.update(accepted=True, revenue=self.current_request.get('revenue', 10.0))
            # 注册服务以便后续离开时回收
            self.event_handler.register_service(self.current_request['id'], self.current_tree)
        else:
            self.metrics.update(accepted=False)

        # 时间推进与离开事件
        leaves = self.data_loader.get_current_leaves()
        self.event_handler.process_leaves(leaves)
        self.data_loader.advance_time()

        # 下一个请求
        if req_done:
            self._reset_current_request()

        return self.get_state(), reward, sub_done, req_done or self.phase_done, info

    # =====================================================================
    # 辅助功能
    # =====================================================================
    def render_failure(self, failed_dest_idx: int, failed_path=None, title: str = "Failure"):
        if not self.enable_render or not self.current_request:
            return

        src = self.current_request['source']
        dests = self.current_request.get('dest', [])
        failed_node = dests[failed_dest_idx] if failed_dest_idx < len(dests) else -1

        success_paths = self.current_tree.get('paths_map', {})
        vnf_placement = {}
        ns, vs = np.where(self.current_tree['hvt'] > 0)
        for n, v in zip(ns, vs):
            vnf_placement[f"VNF_{v}"] = n + 1  # 转为 1-based

        req_id = self.current_request.get('id', self.total_requests_seen)
        filename = f"fail_req{req_id}_goal{failed_dest_idx}_node{failed_node}.png"
        full_path = self.failure_output_dir / filename

        self.visualizer.draw_failure_case(
            src=src,
            dests=dests,
            success_paths=success_paths,
            vnf_placement=vnf_placement,
            failed_dest=failed_node,
            failed_path=failed_path,
            title=title,
            save_path=str(full_path)
        )

    def get_backup_metrics(self) -> Dict[str, float]:
        return self.policy_helper.backup_policy.get_metrics()

    def print_env_summary(self):
        logger.info("=" * 50)
        logger.info("ENVIRONMENT SUMMARY")
        logger.info(f"Time step       : {self.data_loader.time_step}")
        logger.info(f"Requests seen   : {self.total_requests_seen}")
        logger.info(f"Requests accepted: {self.total_requests_accepted} "
                    f"({self.total_requests_accepted / max(1, self.total_requests_seen):.2%})")
        bk = self.get_backup_metrics()
        logger.info(f"Backup Activated: {bk['activation_rate']:.1f}% "
                    f"| Success Rate: {bk['success_rate']:.1f}%")
        logger.info(f"VNF Sharing Rate: {self.resource_mgr.get_vnf_sharing_rate():.3f}")
        logger.info("=" * 50)

    def _state_novelty(self) -> float:
        """基于状态访问次数的新颖度奖励"""
        state = self._get_state()
        if isinstance(state, tuple):
            return 0.0  # GNN 模式暂不计算

        s = tuple(np.round(state, 3))
        if s not in self.state_visit_counter:
            self.state_visit_counter[s] = 0
        self.state_visit_counter[s] += 1
        return 1.0 / np.sqrt(1 + self.state_visit_counter[s])

    # =====================================================================
    # 兼容旧接口（方便直接使用）
    # =====================================================================
    def step(self, action):
        """兼容 gym 标准 step（仅低层）"""
        return self.step_low_level(action)[:4] + ({},)