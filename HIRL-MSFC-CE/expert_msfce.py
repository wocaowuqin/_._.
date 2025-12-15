#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py - OPTIMIZED VERSION WITH DISTANCE MATRIX CACHE
"""
优化版本特性：
1. ✅ 三级缓存系统：路径缓存 + 链路缓存 + 距离矩阵
2. ✅ 预计算所有路径（初始化时一次性完成）
3. ✅ O(1) 路径查询和距离查询
4. ✅ 完整的缓存诊断和验证功能
5. ✅ SolverConfig 配置类
6. ✅ Rollback 机制
7. ✅ 增强 Recall 策略
8. ✅ 完整资源验证（含带宽）
"""

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

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ========== 配置类 ==========
@dataclass
class SolverConfig:
    """集中式配置管理"""
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

    def __post_init__(self):
        """参数验证"""
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("Alpha, beta, gamma must be between 0 and 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            logger.warning("Score weights do not sum to 1.0")


def parse_mat_request(req_obj) -> Dict:
    """解析请求（兼容 Python Dict 和 MATLAB 格式）"""
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
            'arrival_time': int(req_obj.get('arrival_time', [[0]])[0, 0]),
            'leave_time': int(req_obj.get('leave_time', [[0]])[0, 0]),
        }
    except:
        return {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }


class MSFCE_Solver:
    """MSFC-CE 专家算法求解器（优化版）"""

    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict,
                 config: Optional[SolverConfig] = None):

        self.config = config or SolverConfig()

        # 加载 Path DB
        if not Path(path_db_file).exists():
            raise FileNotFoundError(f"Path DB missing: {path_db_file}")

        try:
            mat = sio.loadmat(path_db_file)
            self.path_db = mat['Paths']
            logger.info(f"Loaded Path DB from {path_db_file}")
        except Exception as e:
            raise RuntimeError(f"Path DB load failed: {e}")

        # 网络拓扑
        self.node_num = int(topology_matrix.shape[0])
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        # VNF 类型和 DC 节点
        self.type_num = 8
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        # 资源容量
        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        # K 条路径
        self.k_path = int(self.config.k_path)
        self.k_path_count = self.k_path

        # LRU 缓存（用于路径评分）
        self._path_eval_cache = OrderedDict()
        self.MAX_CACHE_SIZE = int(self.config.max_cache_size)

        # ✅ 新增：三级缓存系统
        self._path_cache = {}  # 一级：完整路径缓存 {(src, dst, k): (nodes, dist, links)}
        self._link_cache = {}  # 二级：链路ID缓存 {(u, v): link_id}
        self._distance_matrix = None  # 三级：距离矩阵缓存 (n x n)

        # 性能指标
        self.metrics = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'failure_reasons': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'errors': 0,
        }

        # ✅ 预计算阶段
        logger.info("=" * 60)
        logger.info("OPTIMIZATION: Starting Path Precomputation")
        logger.info("=" * 60)

        start_total = time.time()

        # Step 1: 构建链路查找表
        self._build_link_lookup()

        # Step 2: 预计算所有路径
        logger.info("Precomputing path database...")
        start_precompute = time.time()
        self._precompute_all_paths()
        elapsed_precompute = time.time() - start_precompute
        logger.info(f"✓ Path cache initialized: {len(self._path_cache)} entries in {elapsed_precompute:.2f}s")

        # Step 3: 预计算距离矩阵
        logger.info("Precomputing distance matrix...")
        start_dist = time.time()
        self._precompute_distance_matrix()
        elapsed_dist = time.time() - start_dist
        logger.info(f"✓ Distance matrix ready in {elapsed_dist:.2f}s")

        elapsed_total = time.time() - start_total
        logger.info(f"✓ Total optimization time: {elapsed_total:.2f}s")

        # Step 4: 验证缓存
        self.validate_cache()

        logger.info("=" * 60)

        # ========== 初始化诊断 ==========
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: Expert MSFCE Initialization")
        logger.info("=" * 60)
        logger.info(f"✓ Node count: {self.node_num}")
        logger.info(f"✓ Link count: {self.link_num}")
        logger.info(f"✓ Type count: {self.type_num}")
        logger.info(f"✓ K-path: {self.k_path}")
        logger.info(f"✓ DC count: {len(self.DC)}")

        if len(self.DC) == 0:
            logger.error("✗ ERROR: DC list is EMPTY!")
        else:
            dc_sorted = sorted(list(self.DC))
            logger.info(f"✓ DC nodes (first 10): {dc_sorted[:10]}")

        logger.info(f"✓ Capacities: CPU={self.cap_cpu}, MEM={self.cap_mem}, BW={self.cap_bw}")
        logger.info("=" * 60)

    # ========== 新增：预计算核心方法 ==========

    def _build_link_lookup(self):
        """
        构建快速链路查找表（O(1)时间复杂度）
        支持双向查询：(u,v) 和 (v,u) 都能找到同一个 link_id
        """
        self._link_cache.clear()

        # 从 self.link_map 复制（已经是双向的）
        for edge, lid in self.link_map.items():
            self._link_cache[edge] = lid

        logger.info(f"✓ Link lookup table built: {len(self._link_cache)} entries")

    def _precompute_all_paths(self):
        """
        预计算所有路径的节点和链路ID
        只在初始化时执行一次，后续查询直接返回
        """
        total_paths = 0
        failed_paths = 0

        for src in range(1, self.node_num + 1):
            for dst in range(1, self.node_num + 1):
                if src == dst:
                    # 自环直接存储
                    self._path_cache[(src, dst, 1)] = ([src], 0, [])
                    total_paths += 1
                    continue

                # 尝试加载前 k_path 条路径
                for k in range(1, self.k_path + 1):
                    try:
                        nodes, dist, links = self._load_path_from_db(src, dst, k)

                        if nodes:  # 只缓存有效路径
                            self._path_cache[(src, dst, k)] = (nodes, dist, links)
                            total_paths += 1
                        else:
                            failed_paths += 1
                    except Exception as e:
                        logger.debug(f"Failed to load path ({src},{dst},k={k}): {e}")
                        failed_paths += 1

        logger.info(f"  - Valid paths: {total_paths}")
        logger.info(f"  - Failed paths: {failed_paths}")

        # 计算覆盖率
        expected_paths = self.node_num * (self.node_num - 1) * self.k_path
        coverage = (total_paths - self.node_num) / expected_paths  # 减去自环
        logger.info(f"  - Coverage: {coverage:.2%}")

    def _load_path_from_db(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """
        从PathDB加载单条路径（仅在预计算时调用）
        这是原 _get_path_info 的精简版，只负责解析数据
        """
        try:
            # 访问路径数据 (转为0-based索引)
            pinfo = self.path_db[src - 1, dst - 1]

            # 检查paths字段
            if 'paths' not in pinfo.dtype.names:
                return [], 0, []

            raw_paths = pinfo['paths']
            if raw_paths.size == 0:
                return [], 0, []

            # 获取第k条路径
            idx = k - 1
            path_arr = None

            # 处理不同的数据结构
            if raw_paths.dtype == 'O':  # 对象数组
                flat_data = raw_paths.flatten()
                if idx < len(flat_data):
                    path_arr = flat_data[idx]
            elif raw_paths.ndim == 2:  # 二维数组
                if idx < raw_paths.shape[0]:
                    path_arr = raw_paths[idx]
            elif raw_paths.ndim == 1 and idx == 0:  # 一维数组
                path_arr = raw_paths

            if path_arr is None:
                return [], 0, []

            # 获取distance信息
            dist_k = 0
            if 'pathsdistance' in pinfo.dtype.names:
                raw_dists = pinfo['pathsdistance'].flatten()
                if idx < len(raw_dists):
                    dist_k = int(raw_dists[idx])

            # 转换为列表并过滤负值
            path_arr_flat = np.array(path_arr).flatten()

            # 先截取到dist_k+1长度
            if dist_k > 0:
                path_segment = path_arr_flat[:dist_k + 1]
            else:
                path_segment = path_arr_flat

            # 过滤负值和0 (MATLAB填充值)
            path_nodes = [int(x) for x in path_segment if int(x) > 0]

            if len(path_nodes) == 0:
                return [], 0, []

            # ✅ 使用快速链路查找（O(1)时间复杂度）
            links = self._compute_links_fast(path_nodes)

            return path_nodes, len(path_nodes) - 1 if len(path_nodes) > 1 else 0, links

        except Exception as e:
            logger.debug(f"[PATH] Exception for [{src}->{dst}], k={k}: {e}")
            return [], 0, []

    def _compute_links_fast(self, path_nodes: List[int]) -> List[int]:
        """
        快速计算路径的链路ID列表（使用预构建的查找表）
        时间复杂度: O(n) 其中n是路径长度
        """
        links = []

        if len(path_nodes) <= 1:
            return links

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]

            # ✅ 直接查表（O(1)）
            link_id = self._link_cache.get((u, v)) or self._link_cache.get((v, u))

            if link_id is not None:
                links.append(link_id)
            else:
                logger.debug(f"[PATH] No link for edge ({u},{v})")

        return links

    def _precompute_distance_matrix(self):
        """
        预计算最短距离矩阵（用于快速距离查询）
        时间复杂度: O(n²)，但只执行一次
        """
        n = self.node_num
        self._distance_matrix = np.full((n, n), 9999, dtype=int)
        np.fill_diagonal(self._distance_matrix, 0)

        computed_count = 0

        for src in range(1, n + 1):
            for dst in range(1, n + 1):
                if src == dst:
                    continue

                # 使用第1条最短路径的距离
                cache_key = (src, dst, 1)
                if cache_key in self._path_cache:
                    _, dist, _ = self._path_cache[cache_key]
                    self._distance_matrix[src - 1, dst - 1] = dist
                    computed_count += 1

        logger.info(f"  - Distance entries: {computed_count}/{n * (n - 1)}")

    # ========== 优化后的路径查询方法 ==========

    def _get_path_info(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """
        获取路径信息（1-based 索引）

        ✅ 优化版：直接从缓存返回，时间复杂度 O(1)

        Args:
            src: 源节点 (1-based)
            dst: 目标节点 (1-based)
            k: 第k条最短路径 (1-based)

        Returns:
            (path_nodes, distance, link_ids) 或 ([], 0, [])
        """
        # ✅ 快速检查
        if self.path_db is None:
            return [], 0, []

        # ✅ 自环处理
        if src == dst:
            return [src], 0, []

        # ✅ 索引范围检查
        if not (1 <= src <= self.node_num and 1 <= dst <= self.node_num):
            logger.warning(f"[PATH] Invalid nodes: src={src}, dst={dst}, valid=[1,{self.node_num}]")
            return [], 0, []

        if not (1 <= k <= self.k_path):
            logger.debug(f"[PATH] Invalid k={k}, valid=[1,{self.k_path}]")
            return [], 0, []

        # ✅ 核心优化：直接从缓存返回（O(1)时间复杂度）
        cache_key = (src, dst, k)

        if cache_key in self._path_cache:
            self.metrics['cache_hits'] += 1
            return self._path_cache[cache_key]
        else:
            # 缓存未命中（理论上不应该发生）
            self.metrics['cache_misses'] += 1
            logger.warning(f"[PATH] Cache miss for ({src},{dst},k={k}), trying fallback...")

            # ✅ Fallback：动态加载（保留原有逻辑作为安全网）
            try:
                nodes, dist, links = self._load_path_from_db(src, dst, k)
                if nodes:
                    self._path_cache[cache_key] = (nodes, dist, links)  # 存入缓存
                    return nodes, dist, links
            except Exception as e:
                logger.error(f"[PATH] Fallback failed: {e}")

            return [], 0, []

    def get_shortest_distance(self, src: int, dst: int) -> int:
        """
        快速获取最短距离（O(1)时间复杂度）

        Args:
            src: 源节点 (1-based)
            dst: 目标节点 (1-based)

        Returns:
            最短跳数，如果无路径则返回 9999
        """
        if self._distance_matrix is None:
            logger.warning("Distance matrix not initialized, computing on-demand...")
            self._precompute_distance_matrix()

        if 1 <= src <= self.node_num and 1 <= dst <= self.node_num:
            return int(self._distance_matrix[src - 1, dst - 1])

        return 9999

    def _get_max_hops(self, src: int, dst: int) -> int:
        """获取最大跳数（使用最慢的第k条路径）"""
        try:
            # 尝试使用最后一条路径的距离
            cache_key = (src, dst, self.k_path)
            if cache_key in self._path_cache:
                _, dist, _ = self._path_cache[cache_key]
                return dist

            # Fallback：使用第一条路径的距离 * 2
            return self.get_shortest_distance(src, dst) * 2
        except:
            return 10

    # ========== 缓存诊断和管理 ==========

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        import sys

        path_cache_mb = sum(sys.getsizeof(v) for v in self._path_cache.values()) / 1024 / 1024
        link_cache_mb = sys.getsizeof(self._link_cache) / 1024 / 1024
        dist_matrix_mb = self._distance_matrix.nbytes / 1024 / 1024 if self._distance_matrix is not None else 0

        return {
            'path_cache_entries': len(self._path_cache),
            'path_cache_mb': path_cache_mb,
            'link_cache_entries': len(self._link_cache),
            'link_cache_mb': link_cache_mb,
            'distance_matrix_computed': self._distance_matrix is not None,
            'distance_matrix_mb': dist_matrix_mb,
            'total_memory_mb': path_cache_mb + link_cache_mb + dist_matrix_mb,
            'cache_hits': self.metrics.get('cache_hits', 0),
            'cache_misses': self.metrics.get('cache_misses', 0),
            'hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
        }

    def validate_cache(self) -> bool:
        """验证缓存完整性"""
        expected_paths = self.node_num * (self.node_num - 1) * self.k_path
        actual_paths = len(self._path_cache) - self.node_num  # 减去自环

        coverage = actual_paths / expected_paths

        if coverage > 0.8:
            logger.info(f"✓ Cache validation PASSED: {coverage:.2%} coverage ({actual_paths}/{expected_paths})")
            return True
        else:
            logger.warning(f"✗ Cache validation WARNING: {coverage:.2%} coverage (expected >80%)")
            return False

    def print_cache_stats(self):
        """打印详细的缓存统计信息"""
        stats = self.get_cache_stats()

        print("\n" + "=" * 60)
        print("CACHE STATISTICS")
        print("=" * 60)
        print(f"Path Cache:     {stats['path_cache_entries']:,} entries ({stats['path_cache_mb']:.2f} MB)")
        print(f"Link Cache:     {stats['link_cache_entries']:,} entries ({stats['link_cache_mb']:.2f} MB)")
        print(
            f"Distance Matrix: {'✓' if stats['distance_matrix_computed'] else '✗'} ({stats['distance_matrix_mb']:.2f} MB)")
        print(f"Total Memory:   {stats['total_memory_mb']:.2f} MB")
        print("-" * 60)
        print(f"Cache Hits:     {stats['cache_hits']:,}")
        print(f"Cache Misses:   {stats['cache_misses']:,}")
        print(f"Hit Rate:       {stats['hit_rate']:.2%}")
        print("=" * 60 + "\n")

    # ========== 原有方法（保持不变）==========

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        """构建链路映射"""
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _normalize_state(self, state: Dict) -> Dict:
        """标准化状态字典"""
        normalized = {}
        normalized['bw'] = state.get('bw', state.get('bandwidth',
                                                     np.full(self.link_num, self.cap_bw)))
        normalized['cpu'] = state.get('cpu', np.full(self.node_num, self.cap_cpu))
        normalized['mem'] = state.get('mem', state.get('memory',
                                                       np.full(self.node_num, self.cap_mem)))
        normalized['hvt'] = state.get('hvt', state.get('hvt_all',
                                                       np.zeros((self.node_num, self.type_num))))
        normalized['bw_ref_count'] = state.get('bw_ref_count',
                                               np.zeros(self.link_num))
        if 'request' in state:
            normalized['request'] = state['request']
        return normalized

    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        """
        计算路径评分（静态部分缓存 + 动态资源）
        """
        if not nodes:
            return 0.0

        cache_key = (src_node, dst_node, tuple(nodes))

        # 缓存查询（LRU）
        if cache_key in self._path_eval_cache:
            term1, term2 = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = (term1, term2)
            self.metrics['cache_hits'] += 1
        else:
            # 静态部分计算
            max_hops = self._get_max_hops(src_node, dst_node)
            current_hops = len(nodes) - 1
            term1 = 1.0 - (current_hops / max(1, max_hops))

            dc_count = sum(1 for n in nodes if n in self.DC)
            term2 = dc_count / max(1, self.dc_num)

            # 存入缓存
            self._path_eval_cache[cache_key] = (term1, term2)
            self.metrics['cache_misses'] += 1

            # LRU 淘汰
            if len(self._path_eval_cache) > self.MAX_CACHE_SIZE:
                self._path_eval_cache.popitem(last=False)

        # 动态部分（实时资源）
        sr_val = 0.0
        for n in nodes:
            if n in self.DC:
                idx = n - 1
                if idx < len(state['cpu']):
                    sr_val += (state['cpu'][idx] + state['mem'][idx]) / \
                              (self.cap_cpu + self.cap_mem)
        for lid in links:
            idx = lid - 1
            if idx < len(state['bw']):
                sr_val += state['bw'][idx] / self.cap_bw

        norm_factor = max(1, len(nodes) + len(links))
        term3 = sr_val / norm_factor

        return float(self.config.alpha * term1 +
                     self.config.beta * term2 +
                     self.config.gamma * term3)

    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict, Dict]:
        """
        尝试部署 VNF
        Returns: (feasible, hvt, placement, resource_delta_or_reason)
        """
        req_vnfs = request.get('vnf', [])
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)

        if len(path_dcs) < len(req_vnfs):
            reason = {
                'type': 'not_enough_dcs',
                'required': len(req_vnfs),
                'available': len(path_dcs),
                'path': path_nodes
            }
            logger.debug(f"Req {request.get('id', '?')}: {reason['type']}")
            return False, existing_hvt, {}, reason

        current_dc_idx = 0
        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False

            # 1. 尝试复用
            for node in path_dcs:
                node_idx = node - 1
                if node_idx < hvt.shape[0] and (v_type - 1) < hvt.shape[1]:
                    if hvt[node_idx, v_type - 1] > 0:
                        placement[v_idx] = node
                        deployed = True
                        break
            if deployed:
                continue

            # 2. 尝试新部署
            start_search = current_dc_idx
            while start_search < len(path_dcs):
                node = path_dcs[start_search]
                node_idx = node - 1
                if node_idx >= len(state['cpu']):
                    start_search += 1
                    continue

                cpu_req = request['cpu_origin'][v_idx]
                mem_req = request['memory_origin'][v_idx]

                curr_cpu = state['cpu'][node_idx] - cpu_delta[node_idx]
                curr_mem = state['mem'][node_idx] - mem_delta[node_idx]

                if curr_cpu >= cpu_req and curr_mem >= mem_req:
                    cpu_delta[node_idx] += cpu_req
                    mem_delta[node_idx] += mem_req
                    hvt[node_idx, v_type - 1] = 1
                    placement[v_idx] = node
                    deployed = True
                    current_dc_idx = start_search + 1
                    break
                else:
                    start_search += 1

            if not deployed:
                reason = {
                    'type': 'resource_shortage',
                    'vnf_idx': v_idx,
                    'vnf_type': v_type,
                    'cpu_required': request['cpu_origin'][v_idx],
                    'mem_required': request['memory_origin'][v_idx],
                    'checked_nodes': [path_dcs[i] for i in range(current_dc_idx, len(path_dcs))]
                }
                logger.debug(f"Req {request.get('id', '?')}: VNF {v_idx} resource_shortage")
                return False, existing_hvt, {}, reason

        resource_delta = {'cpu': cpu_delta, 'mem': mem_delta}
        return True, hvt, placement, resource_delta

    def _evaluate_otv(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        """计算 OTV 成本"""
        node_cost = np.sum(hvt)
        link_cost = np.sum(tree_links)
        return self.config.otv_link_weight * (link_cost / self.config.otv_norm_link) + \
            self.config.otv_node_weight * (node_cost / self.config.otv_norm_node)

    def _validate_resource_deduction(self, state: Dict, resource_delta: Dict,
                                     request: Dict, links_used: Optional[List[int]] = None) -> bool:
        """完整资源验证（含带宽）"""
        cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
        mem_d = resource_delta.get('mem', np.zeros(self.node_num))

        # 形状验证
        if cpu_d.shape != (self.node_num,) or mem_d.shape != (self.node_num,):
            logger.error(f"Shape mismatch: cpu{cpu_d.shape}, mem{mem_d.shape}")
            return False

        # 非负性验证
        if np.any(cpu_d < -1e-10) or np.any(mem_d < -1e-10):
            logger.error("Negative delta detected")
            return False

        # 容量验证
        if np.any(state['cpu'] - cpu_d < -1e-8):
            logger.error(f"CPU violation at nodes: {np.where(state['cpu'] - cpu_d < 0)[0]}")
            return False
        if np.any(state['mem'] - mem_d < -1e-8):
            logger.error(f"MEM violation at nodes: {np.where(state['mem'] - mem_d < 0)[0]}")
            return False

        # 带宽验证
        if links_used:
            bw_required = request.get('bw_origin', 0.0)
            for lid in links_used:
                idx = lid - 1
                if idx < len(state['bw']):
                    if state['bw'][idx] < bw_required - 1e-8:
                        logger.error(f"BW insufficient on link {lid}: {state['bw'][idx]:.4f} < {bw_required:.4f}")
                        return False

        return True

    def _apply_path_to_tree(self, tree_struct, info, request, state,
                            real_deploy=False, resource_delta=None):
        """应用路径到树（带回滚）"""
        nodes = info['nodes']
        links_used = []

        # 更新链路
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if (u, v) in self.link_map:
                lid = self.link_map[(u, v)]
                idx = lid - 1
                links_used.append(lid)
                if idx < len(tree_struct['tree']):
                    if tree_struct['tree'][idx] == 0:
                        tree_struct['tree'][idx] = 1
                        if real_deploy and idx < len(state['bw']):
                            state['bw'][idx] = max(0.0, state['bw'][idx] - request['bw_origin'])

        tree_struct['nodes'].update(nodes)
        tree_struct['paths_map'][nodes[-1]] = nodes

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        # 资源扣减（验证 + 应用）
        if real_deploy and resource_delta:
            ok = self._validate_resource_deduction(state, resource_delta, request, links_used)
            if not ok:
                raise ValueError("Resource deduction validation failed")

            cpu_d = resource_delta['cpu']
            mem_d = resource_delta['mem']
            state['cpu'] = np.maximum(state['cpu'] - cpu_d, 0.0)
            state['mem'] = np.maximum(state['mem'] - mem_d, 0.0)

    def _apply_path_to_tree_with_rollback(self, tree_struct, info, request, state,
                                          real_deploy=False, resource_delta=None) -> bool:
        """Rollback 机制"""
        # 🔥 修复：使用 copy.deepcopy 而不是 .copy()，支持 tuple/list/dict/ndarray
        original_state = copy.deepcopy({
            'cpu': state['cpu'],
            'mem': state['mem'],
            'bw': state['bw']
        })
        original_tree = {
            'tree': tree_struct['tree'].copy(),
            'hvt': tree_struct['hvt'].copy(),
            'paths_map': copy.deepcopy(tree_struct['paths_map']),
            'nodes': set(tree_struct['nodes'])
        }

        try:
            self._apply_path_to_tree(tree_struct, info, request, state,
                                     real_deploy, resource_delta)
            return True
        except Exception as e:
            # 回滚
            state.update(original_state)
            tree_struct['tree'] = original_tree['tree']
            tree_struct['hvt'] = original_tree['hvt']
            tree_struct['paths_map'] = original_tree['paths_map']
            tree_struct['nodes'] = original_tree['nodes']
            logger.error(f"Rollback: {e}")
            return False

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """返回 8 个值（兼容旧接口）"""
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)

        if not nodes:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dst, 0.0, {}

        score = self._calc_path_eval(nodes, links, state, src, dst)
        temp_state = copy.deepcopy(state)
        feasible, new_hvt, placement, _ = self._try_deploy_vnf(
            request, nodes, temp_state, np.zeros((self.node_num, self.type_num)))

        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1

        cost = self._evaluate_otv(request, tree_vec, new_hvt) if feasible else 0.0
        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """Stage 2: 连接新目标到树"""
        state = self._normalize_state(state)
        request = state.get('request')
        if request is None:
            return {'feasible': False}, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                nodes, dist, links = self._get_path_info(conn_node, request['dest'][d_idx], k)

                if not nodes or len(nodes) < 2:
                    continue
                if set(nodes[1:]) & nodes_on_tree:
                    continue

                score = self._calc_path_eval(nodes, links, state, conn_node, request['dest'][d_idx])
                if score > best_eval:
                    temp_state = copy.deepcopy(state)
                    temp_state['request'] = request
                    full_nodes = conn_path[:i_idx + 1] + nodes[1:]
                    existing_hvt = current_tree.get('hvt', np.zeros((self.node_num, self.type_num)))

                    feasible, new_hvt, placement, res_delta = self._try_deploy_vnf(
                        request, full_nodes, temp_state, existing_hvt)

                    if feasible:
                        best_eval = score
                        tree_vec = np.zeros(self.link_num)
                        for lid in links:
                            if lid - 1 < len(tree_vec):
                                tree_vec[lid - 1] = 1
                        best_res = {
                            'tree': tree_vec, 'hvt': new_hvt, 'new_path_full': nodes,
                            'feasible': True, 'placement': placement, 'res_delta': res_delta
                        }
                        best_action = (i_idx, k - 1)

        if best_res:
            cost = self._evaluate_otv(request, best_res['tree'], best_res['hvt'])
            return best_res, best_eval, best_action, cost
        else:
            return {'feasible': False}, 0.0, (0, 0), 0.0

    def _check_resource_feasibility(self, request: Dict, state: Dict) -> bool:
        """快速全局资源检查"""
        total_cpu_req = sum(request.get('cpu_origin', []))
        total_mem_req = sum(request.get('memory_origin', []))
        total_bw_req = request.get('bw_origin', 0.0) * len(request.get('dest', []))

        available_cpu = np.sum(state['cpu'])
        available_mem = np.sum(state['mem'])
        available_bw = np.sum(state['bw'])

        if total_cpu_req > available_cpu or total_mem_req > available_mem:
            return False
        if total_bw_req > available_bw:
            logger.debug("Total BW requirement high (may still succeed due to sharing)")
        return True

    def _get_adaptive_lookahead_depth(self, num_remaining: int) -> int:
        """动态调整 lookahead 深度"""
        if num_remaining <= 2:
            return min(num_remaining, self.config.lookahead_depth)
        elif num_remaining <= 5:
            return min(2, self.config.lookahead_depth)
        else:
            return 1

    def _construct_tree(self, request: Dict, network_state: Dict,
                        forced_first_dest_idx: Optional[int] = None) -> Tuple[Optional[Dict], List]:
        """核心树构建逻辑（含候选集 + Lookahead）"""
        start_time = time.time()
        iteration_count = 0

        current_sim_state = copy.deepcopy(network_state)
        dest_indices = list(range(len(request['dest'])))

        current_tree = {
            'id': request['id'],
            'tree': np.zeros(self.link_num),
            'hvt': np.zeros((self.node_num, self.type_num)),
            'paths_map': {},
            'nodes': {request['source']},
            'added_dest_indices': [],
            'traj': []
        }
        ordered_paths = []

        MAX_CANDIDATES = int(self.config.max_candidates)
        MAX_ITER = int(self.config.max_iterations)
        MAX_TIME = float(self.config.max_time_seconds)

        while len(current_tree['added_dest_indices']) < len(dest_indices):
            iteration_count += 1

            if iteration_count > MAX_ITER:
                logger.warning(f"Req {request['id']}: Max iterations ({MAX_ITER}) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            if time.time() - start_time > MAX_TIME:
                logger.warning(f"Req {request['id']}: Timeout ({MAX_TIME}s) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            unadded = [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            candidates = []

            # A. 候选集生成
            if not ordered_paths:
                # Stage 1: Source -> Dest
                targets = [forced_first_dest_idx] if forced_first_dest_idx is not None else unadded
                for d_idx in targets:
                    if d_idx not in unadded:
                        continue
                    if len(candidates) >= MAX_CANDIDATES:
                        break

                    for k in range(1, self.k_path + 1):
                        score, nodes, t_vec, h_vec, feas, _, cost, pl = self._calc_eval(
                            request, d_idx, k, current_sim_state)

                        if feas:
                            _, _, _, res_delta = self._try_deploy_vnf(
                                request, nodes, current_sim_state,
                                np.zeros((self.node_num, self.type_num)))

                            info = {
                                'nodes': nodes, 'k': k, 'score': score, 'p_idx': 0,
                                'res_delta': res_delta, 'hvt': h_vec, 'tree_vec': t_vec,
                                'placement': pl, 'd_idx': d_idx
                            }
                            candidates.append(info)
            else:
                # Stage 2: Tree -> New Dest
                for p_idx, path in enumerate(ordered_paths):
                    for d_idx in unadded:
                        if len(candidates) >= MAX_CANDIDATES:
                            break

                        res, score, action_in_path, _ = self._calc_atnp(
                            current_tree, path, d_idx, current_sim_state, current_tree['nodes'])

                        if res and res.get('feasible'):
                            k = action_in_path[1] + 1
                            info = {
                                'nodes': res['new_path_full'], 'k': k, 'score': score,
                                'p_idx': p_idx, 'conn_idx_in_path': action_in_path[0],
                                'res_delta': res['res_delta'], 'hvt': res['hvt'],
                                'tree_vec': res['tree'], 'placement': res['placement'],
                                'd_idx': d_idx
                            }
                            candidates.append(info)

            if not candidates:
                return None, unadded

            # B. 候选集排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            candidate_set = candidates[:int(self.config.candidate_set_size)]

            # C. Lookahead 选择
            best_global_otv = float('inf')
            selected_info = None
            current_lookahead_depth = self._get_adaptive_lookahead_depth(len(unadded) - 1)

            for info in candidate_set:
                d_idx = info['d_idx']

                temp_tree_sim = copy.deepcopy(current_tree)
                temp_state_sim = copy.deepcopy(current_sim_state)

                applied = self._apply_path_to_tree_with_rollback(
                    temp_tree_sim, info, request, temp_state_sim,
                    real_deploy=True, resource_delta=info.get('res_delta'))

                if not applied:
                    continue

                # Lookahead
                remaining_after = [d for d in unadded if d != d_idx]
                subsequent_count = 0

                while subsequent_count < current_lookahead_depth and remaining_after:
                    next_candidates = []
                    current_sim_paths = list(temp_tree_sim['paths_map'].values()) \
                        if temp_tree_sim['paths_map'] else [[request['source']]]

                    for next_d_idx in remaining_after:
                        for path in current_sim_paths:
                            res, score, _, _ = self._calc_atnp(
                                temp_tree_sim, path, next_d_idx,
                                temp_state_sim, temp_tree_sim['nodes'])

                            if res and res.get('feasible'):
                                next_candidates.append((score, next_d_idx, res))

                    if not next_candidates:
                        break

                    next_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_next_d, best_res = next_candidates[0]

                    temp_info_next = {
                        'nodes': best_res['new_path_full'],
                        'hvt': best_res['hvt'],
                        'tree_vec': best_res['tree']
                    }

                    applied2 = self._apply_path_to_tree_with_rollback(
                        temp_tree_sim, temp_info_next, request, temp_state_sim,
                        real_deploy=True, resource_delta=best_res['res_delta'])

                    if not applied2:
                        break

                    remaining_after.remove(best_next_d)
                    subsequent_count += 1

                otv = self._evaluate_otv(request, temp_tree_sim['tree'], temp_tree_sim['hvt'])
                if otv < best_global_otv:
                    best_global_otv = otv
                    selected_info = info

            # D. 执行最优选择
            if selected_info:
                d_idx = selected_info['d_idx']

                applied = self._apply_path_to_tree_with_rollback(
                    current_tree, selected_info, request, current_sim_state,
                    real_deploy=True, resource_delta=selected_info.get('res_delta'))

                if not applied:
                    self._record_failure(request.get('id', '?'),
                                         {'type': 'apply_failed', 'info': 'final apply failed'})
                    return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

                current_tree['added_dest_indices'].append(d_idx)
                ordered_paths.append(selected_info['nodes'])

                p_idx = selected_info['p_idx']
                k_idx = selected_info['k'] - 1
                placement = selected_info.get('placement', {})
                action_tuple = (p_idx, k_idx, placement)
                cost = self._evaluate_otv(request, current_tree['tree'], current_tree['hvt'])
                current_tree['traj'].append((d_idx, action_tuple, cost))
            else:
                return None, unadded

        return current_tree, current_tree['traj']

    def _estimate_destination_resource(self, request: Dict, d_idx: int,
                                       network_state: Dict) -> float:
        """估算目标节点的资源需求"""
        cpu = sum(request.get('cpu_origin', []))
        mem = sum(request.get('memory_origin', []))
        bw = request.get('bw_origin', 0.0)
        return float(cpu + mem + bw * 10.0)

    def _enhanced_recall_strategy(self, request: Dict, network_state: Dict,
                                  failed_unadded: List[int]) -> Tuple[Optional[Dict], List]:
        """增强 Recall 策略"""
        if not failed_unadded:
            return None, []

        logger.info(f"Recall for req {request.get('id', '?')} with {len(failed_unadded)} failed dests")

        dest_resources = [(d_idx, self._estimate_destination_resource(request, d_idx, network_state))
                          for d_idx in failed_unadded]
        dest_resources.sort(key=lambda x: x[1])

        for d_idx, _ in dest_resources:
            recall_tree, recall_traj = self._construct_tree(
                request, network_state, forced_first_dest_idx=d_idx)

            if recall_tree is not None:
                logger.info(f"Recall successful using dest {d_idx} first")
                return recall_tree, recall_traj

        return None, []

    def _record_failure(self, request_id, reason_dict):
        """记录失败原因"""
        reason_type = reason_dict.get('type', 'unknown')
        self.metrics['failure_reasons'][reason_type] = \
            self.metrics['failure_reasons'].get(reason_type, 0) + 1

    def clear_cache(self):
        """清空路径评分缓存（不清除预计算缓存）"""
        self._path_eval_cache.clear()
        logger.info("Path evaluation cache cleared")

    def export_metrics(self, path: Optional[Path] = None):
        """导出性能指标到 CSV"""
        import csv
        if path is None:
            path = Path('expert_metrics.csv')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Requests', self.metrics['total_requests']])
            writer.writerow(['Accepted', self.metrics['accepted']])
            writer.writerow(['Rejected', self.metrics['rejected']])

            accept_rate = self.metrics['accepted'] / max(1, self.metrics['total_requests'])
            writer.writerow(['Accept Rate', f"{accept_rate:.2%}"])

            writer.writerow([])
            writer.writerow(['Failure Reason', 'Count'])
            for reason, count in self.metrics.get('failure_reasons', {}).items():
                writer.writerow([reason, count])

            if self.metrics.get('processing_times'):
                writer.writerow([])
                writer.writerow(['Avg Processing Time (s)',
                                 np.mean(self.metrics['processing_times'])])

        logger.info(f"Metrics exported to {path}")

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'total_requests': self.metrics['total_requests'],
            'acceptance_rate': self.metrics['accepted'] / max(1, self.metrics['total_requests']),
            'cache_hit_rate': self.metrics['cache_hits'] /
                              max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'failure_reasons': self.metrics.get('failure_reasons', {}),
        }

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report.update({
                'avg_processing_time': float(np.mean(times)),
                'max_processing_time': float(max(times)),
                'min_processing_time': float(min(times)),
            })

        return report

    def get_detailed_performance_report(self) -> Dict:
        """详细性能报告"""
        report = self.get_performance_report()

        # 缓存效率
        cache_eff = {
            'cache_size': len(self._path_eval_cache),
            'cache_max_size': self.MAX_CACHE_SIZE,
            'cache_utilization': len(self._path_eval_cache) / max(1, self.MAX_CACHE_SIZE)
        }
        report['cache_efficiency'] = cache_eff

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report['recent_performance'] = {
                'last_10_avg': float(np.mean(times[-10:])) if len(times) >= 10 else 0.0,
                'trend': 'improving' if len(times) > 1 and times[-1] < times[0] else 'stable'
            }

        return report

    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        """
        专家算法主入口
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1

        try:
            # 🔥🔥🔥 关键修复：State 类型检查和转换 🔥🔥🔥
            if network_state is None:
                logger.error("[Expert] network_state is None!")
                self.metrics['rejected'] += 1
                return None, []

            # 如果是 tuple 或 list，转换为 dict
            if isinstance(network_state, (tuple, list)):
                logger.warning(f"[Expert] Converting {type(network_state).__name__} to dict")
                if len(network_state) >= 3:
                    import numpy as np
                    network_state = {
                        'cpu': np.array(network_state[0]) if not isinstance(network_state[0], np.ndarray) else
                        network_state[0],
                        'mem': np.array(network_state[1]) if not isinstance(network_state[1], np.ndarray) else
                        network_state[1],
                        'bw': np.array(network_state[2]) if not isinstance(network_state[2], np.ndarray) else
                        network_state[2],
                    }
                else:
                    logger.error(f"[Expert] Invalid state format: tuple/list length < 3")
                    self.metrics['rejected'] += 1
                    return None, []

            # 确保是 dict
            if not isinstance(network_state, dict):
                logger.error(f"[Expert] network_state must be dict, got {type(network_state)}")
                self.metrics['rejected'] += 1
                return None, []

            # 确保资源数组支持 .copy() 方法
            import numpy as np
            for key in ['cpu', 'mem', 'bw']:
                if key in network_state:
                    if isinstance(network_state[key], (tuple, list)):
                        network_state[key] = np.array(network_state[key])
                    elif not isinstance(network_state[key], np.ndarray):
                        network_state[key] = np.array(network_state[key])
            # 🔥🔥🔥 修复结束 🔥🔥🔥

            network_state = self._normalize_state(network_state)
            network_state['request'] = request

            if not self._check_resource_feasibility(request, network_state):
                logger.warning(f"Req {request.get('id', '?')} skipped: insufficient resources")
                self.metrics['rejected'] += 1
                self._record_failure(request.get('id', '?'), {'type': 'global_resource_shortage'})
                return None, []

            res_tree, res_traj = self._construct_tree(request, network_state)

            proc_time = time.time() - start_time
            self.metrics['processing_times'].append(proc_time)

            if res_tree is not None:
                self.metrics['accepted'] += 1
                return res_tree, res_traj

            failed_dests = res_traj
            if failed_dests:
                recall_tree, recall_traj = self._enhanced_recall_strategy(
                    request, network_state, failed_dests)

                if recall_tree is not None:
                    self.metrics['accepted'] += 1
                    return recall_tree, recall_traj

            self.metrics['rejected'] += 1
            self._record_failure(request.get('id', '?'), {'type': 'construct_tree_failed'})
            return None, []

        except Exception as e:
            logger.exception(f"Unexpected error in req {request.get('id', '?')}: {e}")
            self.metrics['errors'] += 1
            self.metrics['rejected'] += 1
            return None, []



if __name__ == "__main__":
    logger.info("Expert MSFCE module loaded (Optimized Version with Distance Matrix Cache)")
    logger.info("Features: Path Cache + Link Cache + Distance Matrix + Full Validation")