"""
==================================================
NFV-Oriented HRL-GNN Resource Management Module
==================================================

本文件定义了 NFV 在线资源编排系统中的资源管理与状态抽象组件，
用于支持基于图神经网络（GNN）与分层强化学习（HRL）的
服务功能链（SFC）部署与多播编排任务。

--------------------------------------------------
文件功能概述
--------------------------------------------------
该模块主要负责：
1. 网络拓扑与多维资源（带宽 / CPU / 内存 / VNF）的统一建模
2. 请求部署与释放过程中的资源状态演化
3. 为强化学习 Agent 构造可学习的状态表示（Flat State）
4. 为 Expert / Backup Policy 提供结构化网络状态
5. 支持 VNF 共享、多播树扩展等高级 NFV 场景

该文件不包含：
- 强化学习策略本身
- 奖励函数设计
- Episode / Step 调度逻辑

--------------------------------------------------
类说明
--------------------------------------------------
"""

import numpy as np
import networkx as nx
import torch
import logging
from typing import Dict, Set, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    ResourceManager（资源管理器）

    --------------------------------------------------
    类职责
    --------------------------------------------------
    ResourceManager 是系统中负责“真实资源状态维护”的核心类，
    对物理网络及其资源进行抽象，并向上层模块提供统一、可靠的接口。

    它在系统中的角色是：
    - 强化学习环境（Env）的底层资源后端
    - Expert / Backup Policy 的资源与拓扑信息提供者
    - Reward Critic 与性能评估模块的状态来源

    --------------------------------------------------
    核心功能
    --------------------------------------------------
    1. 网络与资源建模
       - 节点数量、链路数量与拓扑结构
       - 链路带宽资源及其复用计数
       - 节点 CPU / 内存资源
       - VNF 实例部署矩阵（HVT）

    2. 资源演化管理
       - 根据部署方案扣减带宽与计算资源
       - 请求结束或失败后的资源释放
       - VNF 实例引用计数与共享关系维护

    3. 状态表示构造
       - 构建扁平化状态向量（Flat State），作为 RL Agent 输入
       - 提供结构化网络状态字典，供专家与备份策略评估

    4. 拓扑与辅助计算
       - 最短路径距离矩阵（用于进度与启发式）
       - 多播树结构分析与最近节点搜索

    5. 资源利用评估
       - VNF 共享率统计
       - 树规模与资源占用辅助分析

    --------------------------------------------------
    构造函数参数
    --------------------------------------------------
    Args:
        topo (np.ndarray):
            网络拓扑邻接矩阵（N × N）

        capacities (Dict):
            网络资源容量配置字典，包含：
            - bandwidth: 链路带宽容量
            - cpu: 节点 CPU 容量
            - memory: 节点内存容量

        dc_nodes (List[int]):
            数据中心节点编号集合，用于特征提取与策略判断

    --------------------------------------------------
    对外提供的主要接口
    --------------------------------------------------
    - apply_deployment(...)
      应用请求部署方案并扣减资源

    - release_resources_from_req(...)
      请求结束后释放其占用资源

    - get_flat_state(...)
      构造强化学习所需的扁平化状态向量

    - get_network_state_dict(...)
      返回结构化网络资源状态

    - can_share_vnf(...) / share_vnf(...)
      支持 VNF 共享逻辑

    - reset()
      重置网络资源状态

    --------------------------------------------------
    设计说明
    --------------------------------------------------
    ResourceManager 将资源管理逻辑与决策逻辑彻底解耦，
    使 HRL 与 GNN 模型可以专注于“如何决策”，
    而不必关心底层资源的一致性与正确性。

    该类是整个 NFV-Oriented 在线资源编排系统中
    唯一可信的资源状态来源。
    """

    # envs/modules/resource.py

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        """
        初始化资源管理器
        :param link_map: (可选) 物理链路 ID 映射字典 {(u, v): link_id}
        """
        self.topo = topo
        self.n = topo.shape[0]
        self.L = np.sum(topo > 0)

        # 1. 资源容量配置
        self.B_cap = capacities.get('bandwidth', 80.0)
        self.C_cap = capacities.get('cpu', 60.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.K_vnf = 5

        # 保存 DC 节点集合
        self.dc_nodes = set(dc_nodes)

        # 🔥 [修复] 初始化 link_map，防止 _build_edge_index 报错
        self.link_map = link_map

        # 2. 动态资源状态矩阵
        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 3. 状态向量维度定义
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        # 缓存与 VNF 共享
        self._dest_dist_cache = {}
        self.vnf_sharing_map = {}

        # 构建矩阵与图索引
        self._build_shortest_dist_matrix()

        # [新增] GNN 维度定义
        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 14 + 10

        # 🔥 [修复] 必须在 self.link_map 初始化之后调用
        self._build_edge_index()

    def apply_deployment(self, request: Dict, plan: Dict):
        """应用部署方案，扣除资源"""
        tree_branch = plan.get('tree', np.zeros(self.L))
        hvt_branch = plan.get('hvt', np.zeros((self.n, self.K_vnf)))
        bw_req = float(request.get('bw_origin', 0.0))

        # 扣带宽
        for link_idx in np.where(tree_branch > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                self.B[link_idx] = max(0.0, self.B[link_idx] - bw_req)
            self.link_ref_count[link_idx] += 1

        # 扣计算资源
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            if self.hvt_all[node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(int(vnf_t + 1))
                    self.C[node] = max(0.0, self.C[node] - request['cpu_origin'][j])
                    self.M[node] = max(0.0, self.M[node] - request['memory_origin'][j])
                except:
                    pass
            self.hvt_all[node, vnf_t] += 1

    def _build_shortest_dist_matrix(self):
        """构建最短路矩阵 (简单版，用于Progress计算)"""
        self.shortest_dist = np.full((self.n, self.n), 9999.0)
        np.fill_diagonal(self.shortest_dist, 0.0)
        # 如果有 GNN 需求，这里可以用 networkx 计算真实距离
        # G = nx.from_numpy_array(self.topo)
        # dict_len = dict(nx.all_pairs_dijkstra_path_length(G))
        # ... 填充矩阵

    def can_share_vnf(self, node_id: int, vnf_type: int) -> bool:
        """对应 can_share_vnf (逻辑略作解耦)"""
        node_idx = node_id - 1
        # 如果该位置没有VNF实例,不能共享
        if self.hvt_all[node_idx, vnf_type] == 0:
            return False

        # 检查是否达到共享上限 (例如最多3个)
        key = (node_id, vnf_type)
        if key in self.vnf_sharing_map:
            return len(self.vnf_sharing_map[key]) < 3
        return True

    def find_closest_tree_node(self, nodes_on_tree: set, goal_node: int, source_node: int):
        """对应 _find_closest_tree_node_to_goal"""
        if not nodes_on_tree:
            return source_node

        min_dist = float('inf')
        closest = list(nodes_on_tree)[0]

        for node in nodes_on_tree:
            dist = self.get_shortest_distance(node, goal_node)
            if dist < min_dist:
                min_dist = dist
                closest = node
        return closest

    def get_shortest_distance(self, src: int, dst: int) -> float:
        if src == dst: return 0.0
        s, d = src - 1, dst - 1
        if 0 <= s < self.n and 0 <= d < self.n:
            return float(self.shortest_dist[s, d])
        return 9999.0

    def get_flat_state(self, current_request: Optional[Dict],
                       unadded_dest_indices: Set[int],
                       nodes_on_tree: Set[int],
                       current_tree: Optional[Dict]) -> np.ndarray:
        """
        获取扁平化状态向量 (DRL 输入)

        Args:
            current_request: 当前请求信息 (从 Env 传入)
            unadded_dest_indices: 未完成目标集合 (从 Env 传入)
            nodes_on_tree: 当前树上的节点 (从 Env 传入)
            current_tree: 当前树结构 (从 Env 传入)
        """
        # 1. 获取归一化网络状态
        # CPU/Mem 使用率 (N维)
        cpu_usage = (self.C_cap - self.C) / max(1.0, self.C_cap)
        mem_usage = (self.M_cap - self.M) / max(1.0, self.M_cap)
        # 带宽使用率 (L维)
        bw_usage = (self.B_cap - self.B) / max(1.0, self.B_cap)
        # VNF 部署状态归一化 (N*K维)
        hvt_norm = np.clip(self.hvt_all.flatten() / 10.0, 0, 1)

        # 2. 构建请求特征向量 (10维)
        req_vec = np.zeros(self.dim_request, dtype=np.float32)

        if current_request:
            # [0-3] 基础需求特征
            req_vec[0] = current_request.get('bw_origin', 0.0) / max(1.0, self.B_cap)
            if current_request.get('cpu_origin') is not None:
                req_vec[1] = np.mean(current_request['cpu_origin']) / max(1.0, self.C_cap)
            if current_request.get('memory_origin') is not None:
                req_vec[2] = np.mean(current_request['memory_origin']) / max(1.0, self.M_cap)
            req_vec[3] = len(current_request.get('vnf', [])) / 8.0

            # [4-5] 目标完成进度
            dests = current_request.get('dest', [])
            req_vec[4] = len(dests) / 10.0
            if len(dests) > 0:
                completed = len(dests) - len(unadded_dest_indices)
                req_vec[5] = completed / len(dests)

            # [6] 源节点是否在 DC
            req_vec[6] = 1.0 if current_request.get('source') in self.dc_nodes else 0.0

            # [7-9] 树的状态
            if nodes_on_tree:
                req_vec[7] = len(nodes_on_tree) / max(1, self.n)
            if current_tree:
                req_vec[8] = np.sum(current_tree['tree'] > 0) / max(1.0, self.L)
            req_vec[9] = len(unadded_dest_indices) / max(1, len(dests))

        # 3. 拼接
        flat_net = np.concatenate([cpu_usage, mem_usage, bw_usage, hvt_norm])

        # 确保维度对齐
        final_state = np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)
        len_net = min(len(flat_net), self.dim_network)
        final_state[:len_net] = flat_net[:len_net]
        final_state[self.dim_network:] = req_vec

        return final_state

    # ===============================================================
    # 资源操作方法 (保持不变，供 Env 调用)
    # ===============================================================
    def get_network_state_dict(self, current_request=None):
        """对应 _get_network_state_dict"""
        state = {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }
        if current_request:
            state['request'] = current_request
        return state
    def get_vnf_sharing_rate(self) -> float:
        """对应 get_vnf_sharing_rate"""
        if not self.vnf_sharing_map: return 0.0
        total = sum(len(dests) for dests in self.vnf_sharing_map.values())
        unique = len(self.vnf_sharing_map)
        if total == 0: return 0.0
        return 1.0 - (unique / total)
    def reset(self):
        self.B.fill(self.B_cap)
        self.C.fill(self.C_cap)
        self.M.fill(self.M_cap)
        self.hvt_all.fill(0)
        self.link_ref_count.fill(0)
        self.vnf_sharing_map.clear()
        self._dest_dist_cache.clear()
    def release_resources_from_req(self, req, tree):
        """释放资源 (辅助EventHandler)"""
        bw_ret = float(req.get('bw_origin', 0.0))
        for link_idx in np.where(tree['tree'] > 0)[0]:
            if self.link_ref_count[link_idx] > 0:
                self.link_ref_count[link_idx] -= 1
            if self.link_ref_count[link_idx] == 0:
                self.B[link_idx] = min(self.B_cap, self.B[link_idx] + bw_ret)

        for node, vnf_t in np.argwhere(tree['hvt'] > 0):
            if self.hvt_all[node, vnf_t] > 0:
                self.hvt_all[node, vnf_t] -= 1
                if self.hvt_all[node, vnf_t] == 0:
                    try:
                        j = req['vnf'].index(int(vnf_t + 1))
                        self.C[node] = min(self.C_cap, self.C[node] + req['cpu_origin'][j])
                        self.M[node] = min(self.M_cap, self.M[node] + req['memory_origin'][j])
                    except:
                        pass
    def share_vnf(self, node_id: int, vnf_type: int, dest_idx: int):
        """对应 share_vnf"""
        key = (node_id, vnf_type)
        if key not in self.vnf_sharing_map:
            self.vnf_sharing_map[key] = set()
        self.vnf_sharing_map[key].add(dest_idx)

    def _build_edge_index(self):
        """构建 PyTorch Geometric 所需的 edge_index"""
        rows, cols = np.where(self.topo > 0)
        self.link_id_to_edge_idx = {}
        edge_list = []

        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_list.append([u, v])

            # 尝试映射物理链路ID
            phys_id = -1
            if self.link_map:
                phys_id = self.link_map.get((u + 1, v + 1))
                if phys_id is None:
                    phys_id = self.link_map.get((v + 1, u + 1))

            if phys_id is not None and phys_id > 0:
                self.link_id_to_edge_idx[phys_id - 1] = idx
            elif not self.link_map:
                if idx < self.L:
                    self.link_id_to_edge_idx[idx] = idx

        self.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        self.edge_hops = torch.tensor([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=torch.float32)

    def get_graph_state(self, current_request, nodes_on_tree, current_tree,
                        served_dest_count: int, sharing_strategy: int, nb_high_goals: int):
        """
        获取图状态 (GNN Input)
        """
        if not current_request:
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            req_vec = torch.zeros(self.request_dim)  # 注意：这里维度可能需要根据实际调整
            return x, self.edge_index, edge_attr, req_vec

        # --- 1. 节点特征 ---
        src = current_request['source']
        dest_set = set(current_request.get('dest', []))

        avg_dist = self._compute_dest_distances(dest_set)
        sharing_potential = self._compute_vnf_sharing_potential(dest_set)

        node_feats = []
        for i in range(self.n):
            nid = i + 1
            feat = [
                1.0 - self.C[i] / max(1, self.C_cap),
                1.0 - self.M[i] / max(1, self.M_cap),
                1.0 if nid in self.dc_nodes else 0.0,
                1.0 if nid == src else 0.0,
                1.0 if nid in dest_set else 0.0,
                1.0 if nid in nodes_on_tree else 0.0
            ]
            # 多播增强
            num_nearby = sum(1 for d in dest_set if 0 <= d - 1 < self.n and self.shortest_dist[i, d - 1] <= 3)
            feat.extend([
                num_nearby / max(1, len(dest_set)),
                1.0 - avg_dist[i] / max(1, np.max(avg_dist)) if np.max(avg_dist) > 0 else 0,
                sharing_potential[i]
            ])
            # VNF One-hot
            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float32)

        # --- 2. 边特征 ---
        num_edges = self.edge_index.shape[1]
        edge_attrs = torch.zeros((num_edges, self.edge_feat_dim), dtype=torch.float32)
        tree_links = current_tree.get('tree', np.zeros(self.L)) if current_tree else np.zeros(self.L)

        for phys_idx in range(self.L):
            if phys_idx not in self.link_id_to_edge_idx: continue
            edge_idx = self.link_id_to_edge_idx[phys_idx]

            bw_util = 1.0 - self.B[phys_idx] / max(1, self.B_cap)
            in_tree = 1.0 if tree_links[phys_idx] > 0 else 0.0
            hop = self.edge_hops[edge_idx].item()

            # 共享率计算 (简化版)
            shared_rate = 0.0  # 如需精确计算需传入 paths_map，此处暂略以保持接口简洁
            avg_traffic = bw_util

            edge_attrs[edge_idx] = torch.tensor([bw_util, in_tree, hop, shared_rate, avg_traffic])

        # --- 3. 请求向量 ---
        # 复用 flat state 的一部分
        flat_state = self.get_flat_state(current_request, set(), set(), None)  # 仅获取基础特征
        base_req = flat_state[self.dim_network:]  # 前10维

        num_dests = len(dest_set)
        avg_bw = current_request.get('bandwidth', 0) / max(1, num_dests)

        multicast_feats = [
            num_dests / max(1, nb_high_goals),
            served_dest_count / max(1, num_dests),
            avg_bw / max(1, self.B_cap),
            sharing_strategy / 3.0
        ]

        dest_enc = np.zeros(nb_high_goals)
        for d_idx, dest in enumerate(current_request.get('dest', [])):
            if d_idx < nb_high_goals: dest_enc[d_idx] = 1.0

        req_vec = np.concatenate([base_req[:10], multicast_feats, dest_enc])
        req_vec = torch.tensor(req_vec, dtype=torch.float32)

        return x, self.edge_index, edge_attrs, req_vec

    def _compute_dest_distances(self, dest_set):
        key = frozenset(dest_set)
        if key in self._dest_dist_cache: return self._dest_dist_cache[key]

        avg_dist = np.zeros(self.n)
        if not dest_set: return avg_dist

        for i in range(self.n):
            dists = [self.shortest_dist[i, d - 1] for d in dest_set if 0 <= d - 1 < self.n]
            avg_dist[i] = np.mean(dists) if dists else 999.0

        self._dest_dist_cache[key] = avg_dist
        return avg_dist

    def _compute_vnf_sharing_potential(self, dest_set):
        avg_dist = self._compute_dest_distances(dest_set)
        dist_factor = 1.0 - avg_dist / (np.max(avg_dist) + 1e-5)
        resource_factor = 1.0 - (self.C / max(1, self.C_cap))

        potential = 0.4 * dist_factor + 0.3 * resource_factor  # 简化计算
        return np.clip(potential, 0, 1)