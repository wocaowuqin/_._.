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
        self._recall_failed_req_ids = set()

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

        # VNF 类型和 DC 节点（安全处理）
        self.type_num = 8  # 默认值，可后续动态覆盖
        if dc_nodes and min(dc_nodes) == 0:
            logger.info("[Expert] Converting DC nodes from 0-based to 1-based")
            self.DC = {n + 1 for n in dc_nodes}
        else:
            self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        # 资源容量（标量转为 float）
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

        # ========== 关键修复：初始化资源状态模板（全部 float64）==========
        self.initial_state_template = {
            'cpu': np.full(self.node_num, self.cap_cpu, dtype=np.float64),  # 节点 CPU 容量
            'mem': np.full(self.node_num, self.cap_mem, dtype=np.float64),  # 节点 Mem 容量
            'bw': np.full(self.link_num, self.cap_bw, dtype=np.float64),  # 链路 BW 容量
            'cpu_load': np.zeros(self.node_num, dtype=np.float64),  # 当前 CPU 负载
            'mem_load': np.zeros(self.node_num, dtype=np.float64),  # 当前 Mem 负载
            'bw_load': np.zeros(self.link_num, dtype=np.float64),  # 当前 BW 负载
            'hvt': np.zeros((self.node_num, self.type_num), dtype=np.float64),  # VNF 实例矩阵
        }

        logger.info("✓ Resource state template initialized with float64 precision (prevents dtype casting errors)")
        logger.info("=" * 60)

        # ========== 初始化诊断 ==========
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: Expert MSFCE Initialization Complete")
        logger.info("=" * 60)
        logger.info(f"✓ Node count: {self.node_num}")
        logger.info(f"✓ Link count: {self.link_num}")
        logger.info(f"✓ VNF type count: {self.type_num}")
        logger.info(f"✓ K-path: {self.k_path}")
        logger.info(f"✓ DC nodes: {sorted(list(self.DC)) if self.DC else 'None'}")
        logger.info(f"✓ Capacities: CPU={self.cap_cpu:.1f}, MEM={self.cap_mem:.1f}, BW={self.cap_bw:.1f}")
        logger.info(f"✓ Path cache entries: {len(self._path_cache)}")
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
        src, dst 必须是 1-based 索引
        """
        try:
            # 访问路径数据 (PathDB 是 0-based 数组，所以下标要 -1)
            # 例如：查找节点 1->2，对应数组下标 [0, 1]
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

            # 处理不同的 MATLAB 数据导出结构
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

            # 转换为扁平数组
            path_arr_flat = np.array(path_arr).flatten()

            # --- 长度截断逻辑 ---
            # 获取 distance 信息作为截断依据（如果存在）
            dist_k = -1
            if 'pathsdistance' in pinfo.dtype.names:
                raw_dists = pinfo['pathsdistance'].flatten()
                if idx < len(raw_dists):
                    dist_k = int(raw_dists[idx])

            # 如果有有效的 distance，先按 distance + 1 (节点数) 截取
            # 这样可以避免后面有非零的垃圾数据
            if dist_k >= 0 and (dist_k + 1) <= len(path_arr_flat):
                path_segment = path_arr_flat[:dist_k + 1]
            else:
                path_segment = path_arr_flat

            # --- 0值过滤逻辑 (关键) ---
            # MATLAB 常用 0 做 padding，必须过滤掉
            path_nodes = [int(x) for x in path_segment if int(x) > 0]

            if len(path_nodes) < 2:  # 路径至少要有起点和终点
                return [], 0, []

            # ✅ 验证起点和终点是否匹配 (可选的安全检查)
            # if path_nodes[0] != src or path_nodes[-1] != dst:
            #     # 某些情况下 k-path 可能乱序，这里通常不做强制检查，信任 DB
            #     pass

            # ✅ 使用快速链路查找
            links = self._compute_links_fast(path_nodes)

            return path_nodes, len(path_nodes) - 1, links

        except Exception as e:
            # logger.debug(f"[PATH] Exception for [{src}->{dst}], k={k}: {e}")
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
        norm = {}
        norm['cpu'] = np.array(
            state.get('cpu', np.full(self.node_num, self.cap_cpu)),
            dtype=float
        )
        norm['mem'] = np.array(
            state.get('mem', np.full(self.node_num, self.cap_mem)),
            dtype=float
        )
        norm['bw'] = np.array(
            state.get('bw', np.full(self.link_num, self.cap_bw)),
            dtype=float
        )

        # 🔥 新增：统一资源负载（关键）
        norm['cpu_load'] = np.array(
            state.get('cpu_load', np.zeros(self.node_num)),
            dtype=float
        )
        norm['mem_load'] = np.array(
            state.get('mem_load', np.zeros(self.node_num)),
            dtype=float
        )
        norm['bw_load'] = np.array(
            state.get('bw_load', np.zeros(self.link_num)),
            dtype=float
        )

        if 'request' in state:
            norm['request'] = state['request']
        if 'hvt' in state:
            norm['hvt'] = state['hvt']

        return norm

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

        for node in candidate_nodes:
            idx = node - 1

            # 🔥 正确剩余资源 = 容量 - 已占用 - 本次 delta
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

    def _evaluate_otv(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        """计算 OTV 成本"""
        node_cost = np.sum(hvt)
        link_cost = np.sum(tree_links)
        return self.config.otv_link_weight * (link_cost / self.config.otv_norm_link) + \
            self.config.otv_node_weight * (node_cost / self.config.otv_norm_node)

    def _validate_resource_deduction(self, state: Dict, resource_delta: Dict,
                                     request: Dict, links_used: Optional[List[int]] = None) -> bool:
        logger.debug("Validating resource deduction...")
        """完整资源验证（含带宽）"""
        cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
        mem_d = resource_delta.get('mem', np.zeros(self.node_num))
        logger.debug(f"CPU Delta: {cpu_d}")
        logger.debug(f"Memory Delta: {mem_d}")
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
            logger.debug(f"Checking bandwidth requirement: {bw_required}")
            for lid in links_used:
                idx = lid - 1
                if idx < len(state['bw']):
                    if state['bw'][idx] < bw_required - 1e-8:
                        logger.error(f"BW insufficient on link {lid}: {state['bw'][idx]:.4f} < {bw_required:.4f}")
                        return False

        return True

    def _apply_path_to_tree(self, tree_struct: Dict, info: Dict, request: Dict, state: Dict,
                            real_deploy: bool = False, resource_delta: Dict = None) -> bool:
        """
        应用路径到树（标准版接口：info 包含 nodes/links）
        Signature: (tree, info, req, state, deploy, delta)
        """
        nodes = info['nodes']

        # 1. 识别需要扣除带宽的新链路
        links_to_deduct = []
        new_links_indices = set()

        # 自动计算 links (如果 info 中没有提供)
        path_links = info.get('links')
        if path_links is None:
            path_links = self._compute_links_fast(nodes)

        for lid in path_links:
            idx = lid - 1
            if 0 <= idx < self.link_num:
                # 只有当前不在树上 (0) 时才标记为新链路
                if tree_struct['tree'][idx] == 0 and idx not in new_links_indices:
                    new_links_indices.add(idx)
                    if real_deploy:
                        links_to_deduct.append(lid)

        # 2. 资源验证 (仅真实部署时)
        if real_deploy and resource_delta:
            # 验证带宽 (仅验证新加入的链路)
            # 使用 _check_resource_feasibility_safe 或 _validate_resource_deduction
            # 这里统一用 _check_resource_feasibility_safe (假设您已更新为此名称)
            if hasattr(self, '_check_resource_feasibility_safe'):
                if not self._check_resource_feasibility_safe(state, resource_delta, request, links_to_deduct):
                    return False
            elif hasattr(self, '_validate_resource_deduction'):
                if not self._validate_resource_deduction(state, resource_delta, request, links_to_deduct):
                    return False

        # 3. 更新树结构
        for idx in new_links_indices:
            tree_struct['tree'][idx] = 1

        tree_struct['nodes'].update(nodes)
        tree_struct['paths_map'][nodes[-1]] = nodes

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        # 更新计数器
        if 'link_count' not in tree_struct: tree_struct['link_count'] = 0
        if 'node_count' not in tree_struct: tree_struct['node_count'] = 0
        tree_struct['link_count'] += len(new_links_indices)
        tree_struct['node_count'] = len(tree_struct['nodes'])

        # 4. 实际扣除资源
        if real_deploy and resource_delta:
            state['cpu'] = np.maximum(state['cpu'] - resource_delta['cpu'], 0.0)
            state['mem'] = np.maximum(state['mem'] - resource_delta['mem'], 0.0)

            bw_req = request.get('bw_origin', 0.0)
            if bw_req > 0:
                for lid in links_to_deduct:
                    idx = lid - 1
                    if idx < len(state['bw']):
                        state['bw'][idx] = max(0.0, state['bw'][idx] - bw_req)

        return True

    def _apply_path_to_tree_with_rollback(self, tree_struct, info, request, state,
                                          real_deploy=False, resource_delta=None) -> bool:
        """安全回滚机制（适配新接口）"""
        if not real_deploy:
            return self._apply_path_to_tree(
                tree_struct, info, request, state, real_deploy=False, resource_delta=resource_delta
            )

        # 备份状态
        backup_state = {
            'cpu': state['cpu'].copy(),
            'mem': state['mem'].copy(),
            'bw': state['bw'].copy(),
            'hvt': state['hvt'].copy() if 'hvt' in state else None
        }

        backup_tree = {
            'tree': tree_struct['tree'].copy(),
            'hvt': tree_struct['hvt'].copy(),
            'paths_map': copy.deepcopy(tree_struct['paths_map']),
            'nodes': set(tree_struct['nodes']),
            'link_count': tree_struct.get('link_count', 0),
            'node_count': tree_struct.get('node_count', 0)
        }

        try:
            success = self._apply_path_to_tree(
                tree_struct, info, request, state, real_deploy=True, resource_delta=resource_delta
            )
            if not success:
                raise ValueError("Resource check returned False")
            return True

        except Exception:
            # 恢复状态
            state['cpu'][:] = backup_state['cpu']
            state['mem'][:] = backup_state['mem']
            state['bw'][:] = backup_state['bw']
            if backup_state['hvt'] is not None and 'hvt' in state:
                state['hvt'][:] = backup_state['hvt']

            # 恢复树
            tree_struct['tree'][:] = backup_tree['tree']
            tree_struct['hvt'][:] = backup_tree['hvt']
            tree_struct['paths_map'] = backup_tree['paths_map']
            tree_struct['nodes'] = backup_tree['nodes']
            tree_struct['link_count'] = backup_tree['link_count']
            tree_struct['node_count'] = backup_tree['node_count']

            return False
    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """
        计算单条候选路径的分数和可行性（修复版）
        返回 8 个值（兼容旧接口）
        """
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]

        # 1. 获取路径信息（节点、距离、链路）
        nodes, dist, links = self._get_path_info(src, dst, k)
        if not nodes:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dst, 0.0, {}

        # 2. 计算路径基础分数（跳数、距离等）
        score = self._calc_path_eval(nodes, links, state, src, dst)

        # 3. 准备临时状态用于试部署
        temp_state = copy.deepcopy(state)
        current_hvt = temp_state.get('hvt', np.zeros((self.node_num, self.type_num)))  # 当前已部署 VNF

        # 4. 试部署 VNF（关键修复：补全参数）
        # 注意：这里是“沿路径部署整个 VNF 链”，不是单个 VNF
        # 假设你的 _try_deploy_vnf 支持批量部署整条链
        # 如果不支持，需要循环部署每个 VNF

        # 计算整个链的资源 delta
        chain = request['vnf']  # [vnf1, vnf2, ..., vnfm]
        cpu_reqs = request['cpu_origin']  # [cpu1, cpu2, ...]
        mem_reqs = request['memory_origin']  # [mem1, mem2, ...]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)
        vnf_delta = np.zeros((self.node_num, self.type_num))  # 可选，用于实例检查

        # 简化假设：VNF 按顺序放在路径节点上（你实际逻辑可能更复杂）
        # 这里用一个简单策略：尽量复用已有节点
        placement = {}  # (node_id, vnf_type) -> placed_node
        feasible = True
        new_hvt = current_hvt.copy()

        for j, vnf_type in enumerate(chain):
            vnf_t = vnf_type - 1  # 1-based → 0-based
            cpu_req = cpu_reqs[j]
            mem_req = mem_reqs[j]

            # 尝试在路径节点上找一个可放置的位置（优先已有同类型 VNF 的节点？或任意？）
            placed = False
            for node in nodes:  # 沿路径尝试
                node_idx = node - 1

                # 检查是否已有同类型 VNF（默认不允许重复实例）
                if new_hvt[node_idx, vnf_t] > 0:
                    continue  # 已有一个，不能再放（标准 NFV）

                # 临时 delta
                temp_cpu_delta = cpu_delta.copy()
                temp_mem_delta = mem_delta.copy()
                temp_cpu_delta[node_idx] += cpu_req
                temp_mem_delta[node_idx] += mem_req

                # 调用完整资源检查
                if self._check_resource_load_feasible(
                        cpu_delta=temp_cpu_delta,
                        mem_delta=temp_mem_delta,
                        vnf_delta=np.zeros((self.node_num, self.type_num)),  # 临时不加
                        links_used=links,
                        state=temp_state,
                        bw_req=request['bw_origin']
                ):
                    # 成功放置
                    cpu_delta[node_idx] += cpu_req
                    mem_delta[node_idx] += mem_req
                    new_hvt[node_idx, vnf_t] = 1
                    vnf_delta[node_idx, vnf_t] = 1
                    placement[(node, vnf_type)] = node
                    placed = True
                    break

            if not placed:
                feasible = False
                break

        # 5. 生成 tree_vec
        tree_vec = np.zeros(self.link_num)
        if feasible:
            unique_links = set(links)
            for lid in unique_links:
                idx = lid - 1
                if idx < len(tree_vec):
                    tree_vec[idx] = 1

        # 6. 计算 OTV 成本
        cost = self._evaluate_otv(request, tree_vec, new_hvt) if feasible else 0.0

        # 7. 返回兼容的 8 元组
        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """Stage 2: 连接新目标到树（高性能优化版 - 移除 deepcopy）"""
        # state = self._normalize_state(state) # 这一步通常在外部做过了，这里可以注释掉以省时
        request = state.get('request')
        if request is None:
            return {'feasible': False}, 0.0, (0, 0), 0.0

        dst = request['dest'][d_idx]
        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        # 缓存 HVT 以减少 dict 访问
        existing_hvt = current_tree.get('hvt', np.zeros((self.node_num, self.type_num)))

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                nodes, dist, links = self._get_path_info(conn_node, dst, k)

                if not nodes or len(nodes) < 2:
                    continue
                if set(nodes[1:]) & nodes_on_tree:  # 避免环路
                    continue

                # 快速预估分数，如果显然不如当前最优，直接跳过
                # score = self._calc_path_eval(...)
                # if score <= best_eval: continue

                # 计算分数
                score = self._calc_path_eval(nodes, links, state, conn_node, dst)
                if score <= best_eval and best_res is not None:
                    continue

                # ===== ⚡️ 优化核心：直接计算 Delta，不复制 State ⚡️ =====
                # 我们不需要 temp_state，只需要计算出增量(delta)，然后拿增量去跟 state 对比即可

                # 1. 准备增量容器
                cpu_delta = np.zeros(self.node_num)
                mem_delta = np.zeros(self.node_num)
                vnf_delta = np.zeros((self.node_num, self.type_num))

                # 2. 模拟 VNF 放置 (逻辑保持不变)
                full_nodes = conn_path[:i_idx + 1] + nodes[1:]
                chain = request['vnf']
                cpu_reqs = request['cpu_origin']
                mem_reqs = request['memory_origin']
                placement = {}
                feasible = True

                for j, vnf_type in enumerate(chain):
                    vnf_t = vnf_type - 1
                    placed = False
                    for n in full_nodes:
                        n_idx = n - 1
                        # 逻辑：优先复用，其次放空节点
                        if existing_hvt[n_idx, vnf_t] > 0:
                            placement[(n, vnf_type)] = n
                            placed = True
                            break
                        # 注意：这里放宽了限制，不再要求 np.sum(hvt)==0，只要该类型没被占用且资源够即可
                        # 这样可以允许不同类型的 VNF 共享同一个节点
                        if vnf_delta[n_idx, vnf_t] == 0:
                            cpu_delta[n_idx] += cpu_reqs[j]
                            mem_delta[n_idx] += mem_reqs[j]
                            vnf_delta[n_idx, vnf_t] = 1
                            placement[(n, vnf_type)] = n
                            placed = True
                            break
                    if not placed:
                        feasible = False
                        break

                if not feasible:
                    continue

                # 3. 资源检查 (直接传入 state 和 delta)
                # 🔥 这里不再需要 temp_state
                if not self._check_resource_load_feasible(
                        cpu_delta=cpu_delta,
                        mem_delta=mem_delta,
                        vnf_delta=vnf_delta,
                        links_used=links,
                        state=state,  # 传原始 state
                        bw_req=request['bw_origin']
                ):
                    continue

                # 4. 记录结果
                tree_vec = np.zeros(self.link_num)
                unique_links = set(links)
                for lid in unique_links:
                    if 0 < lid <= self.link_num:  # 边界保护
                        tree_vec[lid - 1] = 1

                new_hvt = existing_hvt.copy()  # 只有确定的结果才需要 copy
                new_hvt += vnf_delta

                best_eval = score
                best_res = {
                    'tree': tree_vec,
                    'hvt': new_hvt,
                    'new_path_full': full_nodes,
                    'feasible': True,
                    'placement': placement,
                    'res_delta': {'cpu': cpu_delta, 'mem': mem_delta, 'vnf': vnf_delta}
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
    def links_from_t_vec(self, t_vec: np.ndarray) -> List[int]:
        """
        从 tree_vec（链路向量）提取被使用的链路 ID 列表（1-based）

        Args:
            t_vec: np.ndarray, shape (link_num,), 值 >0 表示该链路在树中

        Returns:
            List[int]: 链路 ID 列表（1-based，与你的 lid 一致）
        """
        if t_vec is None or len(t_vec) == 0:
            return []

        # 找到 >0 的位置（0-based 索引），然后 +1 转为 1-based ID
        used_indices = np.where(t_vec > 0)[0]  # 返回数组
        link_ids = (used_indices + 1).tolist()  # 转为 1-based list

        return link_ids

    def _construct_tree(
            self,
            request: Dict,
            network_state: Dict,
            forced_first_dest_idx: Optional[int] = None
    ) -> Tuple[Optional[Dict], List, List]:
        """
        Construct multicast tree with Beam Search.
        RETURN (ALWAYS 3 VALUES):
            tree (or None),
            traj (list),
            failed_dest_indices (list)
        """
        import copy
        start_time = time.time()

        # =====================================================
        # 1. Initialization
        # =====================================================
        initial_tree = {
            'id': request['id'],
            'tree': np.zeros(self.link_num),
            'hvt': np.zeros((self.node_num, self.type_num)),
            'paths_map': {},
            'nodes': {request['source']},
            'added_dest_indices': [],
            'traj': [],
            'link_count': 0,
            'node_count': 1
        }

        BEAM_SIZE = 3
        candidate_trees = [(initial_tree, 0.0)]

        dest_indices = list(range(len(request['dest'])))
        final_completed_trees = []

        # =====================================================
        # 2. Beam Search expansion
        # =====================================================
        for step in range(len(dest_indices)):
            if time.time() - start_time > self.config.max_time_seconds:
                break

            next_generation = []

            for curr_tree, curr_score in candidate_trees:
                unadded = [
                    d for d in dest_indices
                    if d not in curr_tree['added_dest_indices']
                ]

                if not unadded:
                    final_completed_trees.append((curr_tree, curr_score))
                    continue

                targets_to_try = unadded
                if step == 0 and forced_first_dest_idx is not None:
                    targets_to_try = [forced_first_dest_idx]

                for d_idx in targets_to_try:
                    res, score, action, cost = self._calc_atnp(
                        curr_tree,
                        list(curr_tree['nodes']),
                        d_idx,
                        network_state,
                        curr_tree['nodes']
                    )

                    if not res['feasible']:
                        continue

                    new_tree = copy.deepcopy(curr_tree)

                    success = self._apply_path_to_tree(
                        new_tree,
                        {'nodes': res['new_path_full'], 'hvt': res['hvt']},
                        request,
                        network_state,
                        real_deploy=False,
                        resource_delta=res['res_delta']
                    )

                    if not success:
                        continue

                    new_tree['added_dest_indices'].append(d_idx)

                    action_tuple = (0, 0, res.get('placement', {}))
                    new_tree['traj'].append((d_idx, action_tuple, res['res_delta']))

                    new_total_score = curr_score + score
                    next_generation.append((new_tree, new_total_score))

            if not next_generation:
                break

            next_generation.sort(key=lambda x: x[1], reverse=True)
            candidate_trees = next_generation[:BEAM_SIZE]

        # =====================================================
        # 3. Final selection
        # =====================================================
        all_candidates = final_completed_trees + candidate_trees

        if not all_candidates:
            # 全部失败
            return None, [], dest_indices.copy()

        all_candidates.sort(
            key=lambda x: (len(x[0]['added_dest_indices']), x[1]),
            reverse=True
        )

        best_tree, _ = all_candidates[0]

        failed_dests = [
            d for d in dest_indices
            if d not in best_tree['added_dest_indices']
        ]

        if failed_dests:
            # ❗未完全覆盖 → 明确失败，返回失败 dest
            return None, best_tree['traj'], failed_dests

        # ✅ 全部成功
        return best_tree, best_tree['traj'], []

    def _estimate_destination_resource(self, request: Dict, d_idx: int,
                                       network_state: Dict) -> float:
        """估算目标节点的资源需求"""
        cpu = sum(request.get('cpu_origin', []))
        mem = sum(request.get('memory_origin', []))
        bw = request.get('bw_origin', 0.0)
        return float(cpu + mem + bw * 10.0)

    def _enhanced_recall_strategy(self, request, state, failed_dests):
        """
        Final fixed recall strategy:
        - Do NOT shuffle dest list
        - Shuffle dest indices (construction order)
        - Prevent infinite recall loop
        """
        import copy
        import random
        logger.info(f"[Expert] Recall for req {request.get('id', 'unknown')}")

        max_attempts = 5
        all_indices = list(range(len(request['dest'])))

        for attempt in range(max_attempts):
            logger.info(
                f"[Expert] Recall attempt {attempt + 1}/{max_attempts} "
                f"for req {request.get('id', 'unknown')} "
                f"with {len(failed_dests)} failed dests"
            )

            temp_state = {
                'cpu': state['cpu'].copy(),
                'mem': state['mem'].copy(),
                'bw': state['bw'].copy(),
            }

            # 🔥 核心修复：只打乱“构建顺序”，不动 dest 本身
            shuffled_indices = all_indices[:]
            random.shuffle(shuffled_indices)

            tree, traj, new_failed = self._construct_tree_with_order(
                request,
                temp_state,
                shuffled_indices
            )

            # Recall 成功
            if tree is not None:
                return tree, traj

        logger.warning("[Expert] Recall failed after max attempts")

        req_id = request.get('id')
        if req_id is not None:
            self._recall_failed_req_ids.add(req_id)

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

    def solve_request_for_expert(
            self,
            request: Dict,
            network_state: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], List]:
        """
        Expert main entry (FINAL FIXED VERSION)
        - Correct failed_dests semantics
        - No recall dead-loop
        - Respect real network state
        """

        start_time = time.time()
        self.metrics['total_requests'] += 1

        try:
            # =====================================================
            # 1. Prepare CURRENT STATE (real remaining resources)
            # =====================================================
            if network_state is not None:
                current_state = {}
                for k, v in network_state.items():
                    if isinstance(v, np.ndarray):
                        current_state[k] = v.astype(np.float64).copy()
                    else:
                        current_state[k] = copy.deepcopy(v)
            else:
                if not hasattr(self, 'initial_state_template'):
                    self.metrics['rejected'] += 1
                    return None, []
                current_state = copy.deepcopy(self.initial_state_template)

            # =====================================================
            # 2. Request ID alignment (0-based → 1-based)
            # =====================================================
            req_internal = copy.deepcopy(request)

            if req_internal['source'] == 0 or any(d == 0 for d in req_internal['dest']):
                req_internal['source'] += 1
                req_internal['dest'] = [d + 1 for d in req_internal['dest']]

            current_state['request'] = req_internal

            # =====================================================
            # 3. Fast resource feasibility check
            # =====================================================
            if not self._check_resource_feasibility(req_internal, current_state):
                self.metrics['rejected'] += 1
                return None, []

            # =====================================================
            # 4. First attempt: normal MSFCE construction
            # =====================================================
            tree, traj, failed_dests = self._construct_tree(req_internal, current_state)

            proc_time = time.time() - start_time
            self.metrics['processing_times'].append(proc_time)

            if tree is not None:
                self.metrics['accepted'] += 1
                return tree, traj

            # =====================================================
            # 5. Recall (ONLY if there are failed destinations)
            # =====================================================
            req_id = req_internal.get('id')

            if failed_dests and req_id not in self._recall_failed_req_ids:
                logger.info(
                    f"[Expert] Recall for req {req_id} with failed dests: {failed_dests}"
                )

                recall_state = {
                    'cpu': current_state['cpu'].copy(),
                    'mem': current_state['mem'].copy(),
                    'bw': current_state['bw'].copy(),
                }

                recall_tree, recall_traj = self._enhanced_recall_strategy(
                    req_internal,
                    recall_state,
                    failed_dests
                )

                if recall_tree is not None:
                    self.metrics['accepted'] += 1
                    return recall_tree, recall_traj

            # =====================================================
            # 6. Final failure (NO infinite retry)
            # =====================================================
            self.metrics['rejected'] += 1
            return None, []

        except Exception as e:
            logger.exception(f"[Expert] Error in solve_request_for_expert: {e}")
            self.metrics['rejected'] += 1
            return None, []

    def _check_resource_load_feasible(
            self,
            cpu_delta: np.ndarray,
            mem_delta: np.ndarray,
            vnf_delta: np.ndarray,
            links_used: List[int],
            state: Dict,
            bw_req: float
    ) -> bool:
        """
        资源可行性检查（修复版：基于剩余容量模型）
        逻辑：如果 (需求 > 剩余容量)，则拒绝。
        """

        # 1. CPU 检查
        # state['cpu'] 是剩余容量。如果 增量 > 剩余 + 误差，则不可行。
        # 注意：这里假设 state['cpu'] 已经是 float64
        rem_cpu = state.get('cpu')
        if rem_cpu is not None:
            if np.any(cpu_delta > rem_cpu + 1e-7):
                # failure_indices = np.where(cpu_delta > rem_cpu + 1e-7)[0]
                # logger.debug(f"CPU fail at nodes {failure_indices}")
                return False

        # 2. Memory 检查 (兼容 'mem' 和 'memory' 键名)
        rem_mem = state.get('mem', state.get('memory'))
        if rem_mem is not None:
            if np.any(mem_delta > rem_mem + 1e-7):
                return False

        # 3. 带宽检查 (兼容 'bw' 和 'bandwidth' 键名)
        rem_bw = state.get('bw', state.get('bandwidth'))
        if rem_bw is not None and links_used:
            # 带宽是标量需求，检查所有涉及的链路
            unique_links = set(links_used)
            for lid in unique_links:
                idx = lid - 1
                # 越界保护
                if idx < 0 or idx >= len(rem_bw):
                    continue

                # 核心逻辑：如果 请求 > 剩余，则失败
                if bw_req > rem_bw[idx] + 1e-7:
                    # logger.debug(f"BW fail at link {lid}: req {bw_req} > rem {rem_bw[idx]}")
                    return False

        return True

    def _construct_tree_with_order(self, request, network_state, dest_order):
        """
        Construct multicast tree following a given destination order.
        Same logic as _construct_tree, only the dest order is controlled.
        RETURN:
            tree or None,
            traj list,
            failed_dest_indices
        """
        import copy

        # =====================================================
        # 1. Initialize tree
        # =====================================================
        tree_struct = self._init_tree_struct()
        tree_struct['id'] = request['id']
        tree_struct['nodes'].add(request['source'])
        tree_struct['node_count'] = 1

        traj = []
        failed = []

        # =====================================================
        # 2. Follow forced destination order
        # =====================================================
        for d_idx in dest_order:
            res, score, action, cost = self._calc_atnp(
                tree_struct,
                list(tree_struct['nodes']),
                d_idx,
                network_state,
                tree_struct['nodes']
            )

            if not res['feasible']:
                failed.append(d_idx)
                continue

            success = self._apply_path_to_tree(
                tree_struct,
                {'nodes': res['new_path_full'], 'hvt': res['hvt']},
                request,
                network_state,
                real_deploy=False,
                resource_delta=res['res_delta']
            )

            if not success:
                failed.append(d_idx)
                continue

            tree_struct['added_dest_indices'].append(d_idx)

            action_tuple = (0, 0, res.get('placement', {}))
            traj.append((d_idx, action_tuple, res['res_delta']))

        # =====================================================
        # 3. Return result
        # =====================================================
        if failed:
            return None, traj, failed

        return tree_struct, traj, []

    def _init_tree_struct(self) -> Dict:
        """
        Initialize an empty multicast tree structure.
        Used by both normal construction and recall construction.
        """
        return {
            'id': None,                      # will be filled later if needed
            'tree': np.zeros(self.link_num),
            'hvt': np.zeros((self.node_num, self.type_num)),
            'paths_map': {},
            'nodes': set(),                  # start empty; source added later
            'added_dest_indices': [],
            'traj': [],
            'link_count': 0,
            'node_count': 0
        }

if __name__ == "__main__":
    logger.info("Expert MSFCE module loaded (Optimized Version with Distance Matrix Cache)")
    logger.info("Features: Path Cache + Link Cache + Distance Matrix + Full Validation")