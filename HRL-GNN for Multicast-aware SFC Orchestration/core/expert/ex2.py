#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce_resource_fixed.py
# MSFCE Expert – Unified Resource Load Model (CPU / MEM / BW)

from __future__ import annotations
import time
import copy
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import scipy.io as sio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert-FIX] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===============================
# Configuration
# ===============================
@dataclass
class SolverConfig:
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    candidate_set_size: int = 8
    lookahead_depth: int = 1
    k_path: int = 5
    max_cache_size: int = 5000
    max_iterations: int = 500
    max_time_seconds: float = 60.0
    max_candidates: int = 30
    otv_link_weight: float = 0.2
    otv_node_weight: float = 0.8
    otv_norm_link: float = 90.0
    otv_norm_node: float = 8.0


def parse_mat_request(req_obj) -> Dict:
    if isinstance(req_obj, dict):
        return req_obj
    try:
        return {
            'id': int(req_obj['id'][0, 0]),
            'source': int(req_obj['source'][0, 0]),
            'dest': [int(d) for d in req_obj['dest'].flatten()],
            'vnf': [int(v) for v in req_obj['vnf'].flatten()],
            'bw_origin': float(req_obj['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
        }
    except Exception:
        return {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }


# ===============================
# MSFCE Solver
# ===============================
class MSFCE_Solver:

    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict,
                 config: Optional[SolverConfig] = None):

        self.config = config or SolverConfig()

        mat = sio.loadmat(path_db_file)
        self.path_db = mat['Paths']

        self.node_num = topology_matrix.shape[0]
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        self.DC = set(dc_nodes)
        self.dc_num = len(self.DC)
        self.type_num = 8

        # Capacities (STATIC)
        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        self.k_path = self.config.k_path

        # Caches
        self._path_cache = {}
        self._path_eval_cache = OrderedDict()
        self.MAX_CACHE_SIZE = self.config.max_cache_size

        self._precompute_paths()

    # ===============================
    # Topology / Path
    # ===============================
    def _create_link_map(self, topo):
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _precompute_paths(self):
        for src in range(1, self.node_num + 1):
            for dst in range(1, self.node_num + 1):
                if src == dst:
                    self._path_cache[(src, dst, 1)] = ([src], 0, [])
                    continue
                pinfo = self.path_db[src - 1, dst - 1]
                if 'paths' not in pinfo.dtype.names:
                    continue
                raw_paths = pinfo['paths'].flatten()
                for k in range(min(self.k_path, len(raw_paths))):
                    nodes = [int(x) for x in np.array(raw_paths[k]).flatten() if int(x) > 0]
                    links = []
                    for i in range(len(nodes) - 1):
                        links.append(self.link_map[(nodes[i], nodes[i + 1])])
                    self._path_cache[(src, dst, k + 1)] = (nodes, len(nodes) - 1, links)

    def _get_path_info(self, src, dst, k):
        return self._path_cache.get((src, dst, k), ([], 0, []))

    # ===============================
    # State normalization (KEY FIX)
    # ===============================
    def _normalize_state(self, state: Dict) -> Dict:
        norm = {}
        norm['cpu'] = np.array(state.get('cpu', np.full(self.node_num, self.cap_cpu)), dtype=float)
        norm['mem'] = np.array(state.get('mem', np.full(self.node_num, self.cap_mem)), dtype=float)
        norm['bw'] = np.array(state.get('bw', np.full(self.link_num, self.cap_bw)), dtype=float)

        # 🔥 Unified LOAD vectors
        norm['cpu_load'] = np.array(state.get('cpu_load', np.zeros(self.node_num)), dtype=float)
        norm['mem_load'] = np.array(state.get('mem_load', np.zeros(self.node_num)), dtype=float)
        norm['bw_load'] = np.array(state.get('bw_load', np.zeros(self.link_num)), dtype=float)

        norm['hvt'] = state.get('hvt', np.zeros((self.node_num, self.type_num)))
        if 'request' in state:
            norm['request'] = state['request']
        return norm
    # ===============================
    # Unified Resource Feasibility Check
    # ===============================
    def _check_resource_load_feasible(
        self,
        cpu_delta: np.ndarray,
        mem_delta: np.ndarray,
        links_used: List[int],
        state: Dict,
        bw_req: float
    ) -> bool:
        """
        Unified feasibility check:
        CPU / MEM: node load
        BW: link load
        """
        # CPU / MEM
        for i in range(self.node_num):
            if state['cpu_load'][i] + cpu_delta[i] > state['cpu'][i] + 1e-9:
                return False
            if state['mem_load'][i] + mem_delta[i] > state['mem'][i] + 1e-9:
                return False

        # BW
        for lid in links_used:
            idx = lid - 1
            if state['bw_load'][idx] + bw_req > state['bw'][idx] + 1e-9:
                return False

        return True

    # ===============================
    # VNF Deployment (FIXED CPU / MEM)
    # ===============================
    def _try_deploy_vnf(
        self,
        vnf_id: int,
        candidate_nodes: List[int],
        state: Dict,
        cpu_req: float,
        mem_req: float,
        cpu_delta: np.ndarray,
        mem_delta: np.ndarray
    ) -> Optional[int]:
        """
        Try deploy ONE VNF instance on candidate nodes.
        Only update delta, NEVER touch state load here.
        """
        for node in candidate_nodes:
            idx = node - 1

            # 🔥 FIX: remaining = capacity - load - delta
            remain_cpu = (
                state['cpu'][idx]
                - state['cpu_load'][idx]
                - cpu_delta[idx]
            )
            remain_mem = (
                state['mem'][idx]
                - state['mem_load'][idx]
                - mem_delta[idx]
            )

            if remain_cpu + 1e-9 >= cpu_req and remain_mem + 1e-9 >= mem_req:
                cpu_delta[idx] += cpu_req
                mem_delta[idx] += mem_req
                return node

        return None

    # ===============================
    # Apply Path to Tree (FIXED ALL RESOURCES)
    # ===============================
    def _apply_path_to_tree(
        self,
        tree_struct: Dict,
        nodes: List[int],
        links: List[int],
        resource_delta: Dict[str, np.ndarray],
        state: Dict,
        request: Dict,
        real_deploy: bool = False
    ):
        """
        Apply path expansion to multicast tree.
        All resource loads are applied ONLY here.
        """

        if real_deploy:
            # 🔥 Unified resource check (ONE place)
            feasible = self._check_resource_load_feasible(
                resource_delta['cpu'],
                resource_delta['mem'],
                links,
                state,
                request['bw_origin']
            )
            if not feasible:
                raise ValueError("Unified resource infeasible at apply stage")

            # Apply CPU / MEM load
            state['cpu_load'] += resource_delta['cpu']
            state['mem_load'] += resource_delta['mem']

            # Apply BW load (per-link)
            for lid in links:
                state['bw_load'][lid - 1] += request['bw_origin']

        # Tree structure update (kept for OTV / recall)
        for lid in links:
            idx = lid - 1
            if tree_struct['tree'][idx] == 0:
                tree_struct['tree'][idx] = 1
                tree_struct['link_count'] += 1

        for node in nodes:
            nidx = node - 1
            if tree_struct['node'][nidx] == 0:
                tree_struct['node'][nidx] = 1
                tree_struct['node_count'] += 1

    # ===============================
    # Path Evaluation (unchanged)
    # ===============================
    def _calc_path_eval(
        self,
        nodes: List[int],
        links: List[int],
        state: Dict,
        src_node: int,
        dst_node: int
    ) -> float:

        if not nodes:
            return 0.0

        cache_key = (src_node, dst_node, tuple(nodes))
        if cache_key in self._path_eval_cache:
            return self._path_eval_cache[cache_key]

        # Link cost
        link_cost = 0.0
        for lid in links:
            idx = lid - 1
            remain = max(
                0.0,
                state['bw'][idx] - state['bw_load'][idx]
            )
            link_cost += (1.0 / (remain + 1e-6))

        # Node cost
        node_cost = 0.0
        for node in nodes:
            idx = node - 1
            remain_cpu = state['cpu'][idx] - state['cpu_load'][idx]
            remain_mem = state['mem'][idx] - state['mem_load'][idx]
            node_cost += 1.0 / (remain_cpu + remain_mem + 1e-6)

        score = (
            self.config.otv_link_weight * link_cost +
            self.config.otv_node_weight * node_cost
        )

        # Cache
        self._path_eval_cache[cache_key] = score
        if len(self._path_eval_cache) > self.MAX_CACHE_SIZE:
            self._path_eval_cache.popitem(last=False)

        return score
    # ===============================
    # Tree Construction (unchanged logic)
    # ===============================
    def _construct_tree(
        self,
        request: Dict,
        state: Dict,
        real_deploy: bool = False
    ) -> Optional[Dict]:

        src = request['source']
        dests = request['dest']

        tree_struct = {
            'tree': np.zeros(self.link_num, dtype=int),
            'node': np.zeros(self.node_num, dtype=int),
            'link_count': 0,
            'node_count': 0,
            'vnf_place': [],
        }

        state = self._normalize_state(state)

        cpu_delta = np.zeros(self.node_num, dtype=float)
        mem_delta = np.zeros(self.node_num, dtype=float)

        current_src = src

        # ===== Deploy VNFs sequentially =====
        for idx, vnf in enumerate(request['vnf']):
            cpu_req = request['cpu_origin'][idx]
            mem_req = request['memory_origin'][idx]

            candidate_nodes = list(self.DC)

            deploy_node = self._try_deploy_vnf(
                vnf,
                candidate_nodes,
                state,
                cpu_req,
                mem_req,
                cpu_delta,
                mem_delta
            )

            if deploy_node is None:
                return None

            tree_struct['vnf_place'].append(deploy_node)
            current_src = deploy_node

        # ===== Connect destinations =====
        for dst in dests:
            best_score = float('inf')
            best_path = None

            for k in range(1, self.k_path + 1):
                nodes, _, links = self._get_path_info(current_src, dst, k)
                if not nodes:
                    continue

                score = self._calc_path_eval(
                    nodes,
                    links,
                    state,
                    current_src,
                    dst
                )

                if score < best_score:
                    best_score = score
                    best_path = (nodes, links)

            if best_path is None:
                return None

            nodes, links = best_path

            resource_delta = {
                'cpu': cpu_delta,
                'mem': mem_delta
            }

            try:
                self._apply_path_to_tree(
                    tree_struct,
                    nodes,
                    links,
                    resource_delta,
                    state,
                    request,
                    real_deploy=real_deploy
                )
            except ValueError:
                return None

        return tree_struct

    # ===============================
    # Lookahead (kept)
    # ===============================
    def _lookahead(
        self,
        request: Dict,
        state: Dict
    ) -> bool:
        tree = self._construct_tree(request, state, real_deploy=False)
        return tree is not None

    # ===============================
    # Recall (kept)
    # ===============================
    def _recall(
        self,
        request: Dict,
        state: Dict
    ) -> bool:
        return self._lookahead(request, state)

    # ===============================
    # Main Solve Interface
    # ===============================
    def solve_request_for_expert(
        self,
        request: Dict,
        state: Dict
    ) -> bool:
        """
        Main entry for expert MSFCE solver.
        """
        start_time = time.time()
        state = self._normalize_state(state)

        if not self._lookahead(request, state):
            return False

        tree = self._construct_tree(
            request,
            state,
            real_deploy=True
        )

        if tree is None:
            return False

        # Deployment successful
        return True
