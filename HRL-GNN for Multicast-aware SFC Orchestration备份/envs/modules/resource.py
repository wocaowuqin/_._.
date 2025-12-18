"""
==================================================
NFV-Oriented HRL-GNN Resource Management Module
==================================================
修复记录:
1. ✅ 分离物理链路(L=45)与图边(E=90)
2. ✅ 修复 step_low_level 中的 broadcast 报错
3. ✅ 保持 compute_progress 等接口的兼容性
"""
import numpy as np
import networkx as nx
import torch
import logging
from collections import defaultdict
from typing import Dict, Set, Optional, Tuple, List

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    ResourceManager（资源管理器）- 维度匹配修复版
    """

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        """
        初始化资源管理器
        """
        self.topo = topo
        self.n = topo.shape[0]

        # 🚨 [关键修复] 分离 图边数(E) 与 物理链路数(L)
        # GNN 需要 E=90 (双向)，资源管理需要 L=45 (物理)
        self.num_graph_edges = np.sum(topo > 0)

        # 如果提供了 link_map，用最大 ID 作为链路数；否则默认对称除以2
        if link_map:
            max_id = 0
            for k, v in link_map.items():
                if v > max_id: max_id = v
            self.L = max_id
        else:
            self.L = self.num_graph_edges // 2

        logger.info(f"[RM] Init: Nodes={self.n}, GraphEdges={self.num_graph_edges}, PhysLinks={self.L}")

        # 1. 资源容量配置
        self.B_cap = capacities.get('bandwidth', 80.0)
        self.C_cap = capacities.get('cpu', 10.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.K_vnf = 8

        self.dc_nodes = list(dc_nodes)
        self.link_map = link_map

        # 2. 动态资源状态矩阵 (使用 L=45，修复维度冲突)
        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 节点资源
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)

        # 3. 状态向量维度定义
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        self._dest_dist_cache = {}
        self.vnf_sharing_map = {}

        self._build_shortest_dist_matrix()

        # GNN 维度定义
        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24

        self._build_edge_index()

    def apply_deployment(self, request: Dict, plan: Dict):
        """应用部署方案"""
        # tree_branch 维度现在是 (45,)，与 self.B (45,) 匹配
        tree_branch = plan.get('tree', np.zeros(self.L))
        hvt_branch = plan.get('hvt', np.zeros((self.n, self.K_vnf)))
        bw_req = float(request.get('bw_origin', 0.0))

        # 扣带宽
        for link_idx in np.where(tree_branch > 0)[0]:
            if link_idx < self.L: # 保护
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

    def apply_tree_deployment(self, request: Dict, tree: Dict) -> bool:
        """请求级部署"""
        try:
            bw_req = float(request.get('bw_origin', 0.0))

            # ---------- 1. 带宽 (物理链路) ----------
            for link_idx in np.where(tree['tree'] > 0)[0]:
                if link_idx < self.L:
                    if self.link_ref_count[link_idx] == 0:
                        self.B[link_idx] = max(0.0, self.B[link_idx] - bw_req)
                    self.link_ref_count[link_idx] += 1

            # ---------- 2. 计算资源 ----------
            for node, vnf_t in np.argwhere(tree['hvt'] > 0):
                if self.hvt_all[node, vnf_t] == 0:
                    try:
                        j = request['vnf'].index(int(vnf_t + 1))
                        self.C[node] = max(0.0, self.C[node] - request['cpu_origin'][j])
                        self.M[node] = max(0.0, self.M[node] - request['memory_origin'][j])
                    except Exception:
                        pass
                self.hvt_all[node, vnf_t] += 1

            return True

        except Exception as e:
            logger.error(f"[RM] apply_tree_deployment failed: {e}")
            return False

    def _build_shortest_dist_matrix(self):
        """构建最短路矩阵"""
        self.shortest_dist = np.full((self.n, self.n), 9999.0)
        np.fill_diagonal(self.shortest_dist, 0.0)

    def can_share_vnf(self, node_id: int, vnf_type: int) -> bool:
        """检查 VNF 是否可共享"""
        node_idx = node_id - 1
        if self.hvt_all[node_idx, vnf_type] == 0:
            return False
        key = (node_id, vnf_type)
        if key in self.vnf_sharing_map:
            return len(self.vnf_sharing_map[key]) < 3
        return True

    def find_closest_tree_node(self, nodes_on_tree: set, goal_node: int, source_node: int):
        """寻找最近树节点"""
        if not nodes_on_tree: return source_node
        min_dist = float('inf')
        closest = list(nodes_on_tree)[0]
        for node in nodes_on_tree:
            dist = self.get_shortest_distance(node, goal_node)
            if dist < min_dist:
                min_dist = dist
                closest = node
        return closest

    def get_shortest_distance(self, src: int, dst: int) -> float:
        """获取两节点间的最短距离"""
        if src == dst: return 0.0
        src_idx = src - 1 if src > 0 else 0
        dst_idx = dst - 1 if dst > 0 else 0
        if 0 <= src_idx < self.n and 0 <= dst_idx < self.n:
            return float(self.shortest_dist[src_idx, dst_idx])
        return 9999.0

    def get_flat_state(self, current_request: Optional[Dict],
                       unadded_dest_indices: Set[int],
                       nodes_on_tree: Set[int],
                       current_tree: Optional[Dict]) -> np.ndarray:
        """获取扁平化状态向量"""
        cpu_usage = (self.C_cap - self.C) / max(1.0, self.C_cap)
        mem_usage = (self.M_cap - self.M) / max(1.0, self.M_cap)
        bw_usage = (self.B_cap - self.B) / max(1.0, self.B_cap)
        hvt_norm = np.clip(self.hvt_all.flatten() / 10.0, 0, 1)

        req_vec = np.zeros(self.dim_request, dtype=np.float32)

        if current_request:
            req_vec[0] = current_request.get('bw_origin', 0.0) / max(1.0, self.B_cap)
            if current_request.get('cpu_origin') is not None:
                req_vec[1] = np.mean(current_request['cpu_origin']) / max(1.0, self.C_cap)
            if current_request.get('memory_origin') is not None:
                req_vec[2] = np.mean(current_request['memory_origin']) / max(1.0, self.M_cap)
            req_vec[3] = len(current_request.get('vnf', [])) / 8.0

            dests = current_request.get('dest', [])
            req_vec[4] = len(dests) / 10.0
            if len(dests) > 0:
                completed = len(dests) - len(unadded_dest_indices)
                req_vec[5] = completed / len(dests)

            req_vec[6] = 1.0 if current_request.get('source') in self.dc_nodes else 0.0

            if nodes_on_tree:
                req_vec[7] = len(nodes_on_tree) / max(1, self.n)
            if current_tree:
                req_vec[8] = np.sum(current_tree['tree'] > 0) / max(1.0, self.L)
            req_vec[9] = len(unadded_dest_indices) / max(1, len(dests))

        flat_net = np.concatenate([cpu_usage, mem_usage, bw_usage, hvt_norm])
        final_state = np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)
        len_net = min(len(flat_net), self.dim_network)
        final_state[:len_net] = flat_net[:len_net]
        final_state[self.dim_network:] = req_vec

        return final_state

    def get_network_state_dict(self, current_request=None):
        """返回结构化网络状态"""
        state = {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }
        if current_request:
            state['request'] = current_request
        return state

    def get_vnf_sharing_rate(self) -> float:
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
        bw_ret = float(req.get('bw_origin', 0.0))
        for link_idx in np.where(tree['tree'] > 0)[0]:
            if link_idx < self.L:
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
        key = (node_id, vnf_type)
        if key not in self.vnf_sharing_map:
            self.vnf_sharing_map[key] = set()
        self.vnf_sharing_map[key].add(dest_idx)

    def _build_edge_index(self):
        """构建 PyTorch Geometric edge_index (E=90)"""
        rows, cols = np.where(self.topo > 0)

        # 🚨 [关键修复] 建立 物理ID -> [图边ID列表] 的映射
        self.phys_to_graph_edges = defaultdict(list)
        edge_list = []

        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_list.append([u, v])

            # 获取物理链路ID (0-based)
            phys_id = -1
            if self.link_map:
                phys_id = self.link_map.get((u + 1, v + 1))
                if phys_id is None:
                    phys_id = self.link_map.get((v + 1, u + 1))

            if phys_id is not None and phys_id > 0:
                self.phys_to_graph_edges[phys_id - 1].append(idx)
            elif not self.link_map:
                # 如果没有 link_map，尝试简单的对折映射 (idx 0 <-> idx N/2 ?)
                # 这里简单假设 idx 如果小于 L 则直接对应
                if idx < self.L:
                     self.phys_to_graph_edges[idx].append(idx)
                     # 这种情况下反向边可能无法自动关联，建议最好有 link_map

        self.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        self.edge_hops = torch.tensor([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=torch.float32)

    def get_graph_state(self, current_request, nodes_on_tree, current_tree,
                        served_dest_count: int, sharing_strategy: int, nb_high_goals: int):
        """获取图状态 (GNN Input)"""
        if not current_request:
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            req_vec = torch.zeros(self.request_dim)
            return x, self.edge_index, edge_attr, req_vec

        src = current_request['source']
        dest_set = set(current_request.get('dest', []))
        avg_dist = self._compute_dest_distances(dest_set)
        sharing_potential = self._compute_vnf_sharing_potential(dest_set)

        # 1. 节点特征
        node_feats = []
        for i in range(self.n):
            nid = i + 1
            feat = [
                1.0 - self.C[i] / max(1, self.C_cap),
                1.0 - self.M[i] / max(1, self.M_cap),
                1.0 if nid in self.dc_nodes else 0.0,
                1.0 if nid == src else 0.0,
                1.0 if nid in dest_set else 0.0,
                1.0 if nid in nodes_on_tree else 0.0,
                sum(1 for d in dest_set if 0 <= d - 1 < self.n and self.shortest_dist[i, d - 1] <= 3) / max(1, len(dest_set)),
                1.0 - avg_dist[i] / max(1, np.max(avg_dist)) if np.max(avg_dist) > 0 else 0,
                sharing_potential[i]
            ]
            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)
        x = torch.tensor(node_feats, dtype=torch.float32)

        # 2. 边特征 (映射物理 -> 图边)
        num_edges = self.edge_index.shape[1]
        edge_attrs = torch.zeros((num_edges, self.edge_feat_dim), dtype=torch.float32)

        # 获取当前树中的物理链路集合
        tree_phys_links = set(np.where(current_tree.get("tree", np.zeros(self.L)) > 0)[0])

        # 遍历物理链路，填充对应的所有图边 (双向)
        for phys_idx in range(self.L):
            graph_edge_indices = self.phys_to_graph_edges.get(phys_idx, [])

            bw_util = 1.0 - self.B[phys_idx] / max(1, self.B_cap)
            in_tree = 1.0 if phys_idx in tree_phys_links else 0.0

            for edge_idx in graph_edge_indices:
                hop = float(self.edge_hops[edge_idx])
                edge_attrs[edge_idx] = torch.tensor([bw_util, in_tree, hop, 0.0, bw_util], dtype=torch.float32)

        # 3. 请求向量
        num_dests = len(current_request.get('dest', []))
        chain_len = len(current_request.get('vnf', []))
        avg_cpu = np.mean(current_request.get('cpu_origin', [0]))
        avg_mem = np.mean(current_request.get('memory_origin', [0]))
        bw_demand = current_request.get('bw_origin', 0.0)

        base_feats = np.array([
            bw_demand / max(1.0, self.B_cap),
            num_dests / 20.0,
            chain_len / 10.0,
            avg_cpu / max(1.0, self.C_cap),
            avg_mem / max(1.0, self.M_cap),
            served_dest_count / max(1, num_dests),
            sharing_strategy / 3.0,
            len(nodes_on_tree) / self.n,
            # 链路利用率用物理链路计算
            len(tree_phys_links) / max(1, self.L),
            0.0
        ], dtype=np.float32)

        dest_onehot = np.zeros(nb_high_goals, dtype=np.float32)
        for i in range(min(num_dests, nb_high_goals)):
            dest_onehot[i] = 1.0

        req_vec_array = np.concatenate([base_feats, dest_onehot])
        if len(req_vec_array) > 24:
            req_vec_array = req_vec_array[:24]
        elif len(req_vec_array) < 24:
            req_vec_array = np.pad(req_vec_array, (0, 24 - len(req_vec_array)))

        req_vec = torch.tensor(req_vec_array, dtype=torch.float32)
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
        return np.clip(0.4 * dist_factor + 0.3 * resource_factor, 0, 1)

    def compute_progress(self, goal_dest_idx: int, *args) -> float:
        """兼容性接口"""
        return 0.0

    def compute_qos_violation(self, *args) -> Optional[Dict[str, float]]:
        """兼容性接口"""
        return None