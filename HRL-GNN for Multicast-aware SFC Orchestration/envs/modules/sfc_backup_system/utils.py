#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC Backup System Utils
包含所有辅助函数
"""

import numpy as np
from typing import Dict, List, Any, Union
import logging

logger = logging.getLogger("Expert")


def ensure_list(value: Any) -> List:
    """
    确保值是列表类型

    Args:
        value: 任意类型的值

    Returns:
        列表形式的值
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    # 单个值包装成列表
    return [value]


def build_hvt_from_placement(placement: Dict[int, int], n: int, K_vnf: int) -> np.ndarray:
    """
    从 placement 字典构建 HVT 矩阵

    Args:
        placement: {vnf_key: node_idx} 形式的字典
                  vnf_key 可能是 "vnf_0_type_3" 或整数
        n: 节点数量
        K_vnf: VNF 类型数量

    Returns:
        shape (n, K_vnf) 的 float32 数组

    🔥 修复：确保返回 float32 类型，避免 dtype casting 错误
    """
    # ✅ 明确指定 dtype=np.float32
    hvt = np.zeros((n, K_vnf), dtype=np.float32)

    if not placement:
        return hvt

    for vnf_key, node in placement.items():
        # 解析 vnf_key (格式: "vnf_0_type_3" 或 整数)
        try:
            if isinstance(vnf_key, str):
                # 格式: "vnf_{idx}_type_{vnf_type}"
                parts = vnf_key.split('_')
                if len(parts) >= 4:
                    vnf_idx = int(parts[1])
                    vnf_type = int(parts[3])
                else:
                    logger.warning(f"Invalid vnf_key format: {vnf_key}")
                    continue
            elif isinstance(vnf_key, int):
                # 如果是整数，假设是 vnf_type
                vnf_type = vnf_key
            else:
                logger.warning(f"Unknown vnf_key type: {type(vnf_key)}")
                continue

            # 边界检查
            node_idx = int(node)
            if 0 <= node_idx < n and 0 <= vnf_type < K_vnf:
                hvt[node_idx, vnf_type] = 1.0  # ✅ 使用 float 字面量
            else:
                logger.warning(
                    f"Invalid placement: node={node_idx} (max {n - 1}), "
                    f"vnf_type={vnf_type} (max {K_vnf - 1})"
                )

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse vnf_key '{vnf_key}': {e}")
            continue

    # ✅ 最后再次确认类型
    return hvt.astype(np.float32)


def build_tree_vec(tree: Dict[str, float], n: int) -> np.ndarray:
    """
    从树字典构建树向量/矩阵

    Args:
        tree: {edge_key: flow} 形式的字典
              edge_key 可能是 "0-1" 或 "(0, 1)" 或 tuple (0, 1)
        n: 节点数量

    Returns:
        shape (n, n) 的 float32 邻接矩阵，值为流量
    """
    # 初始化零矩阵
    tree_mat = np.zeros((n, n), dtype=np.float32)

    if not tree:
        return tree_mat

    for edge_key, flow in tree.items():
        try:
            # 解析 edge_key
            if isinstance(edge_key, str):
                # 格式: "0-1" 或 "(0, 1)"
                edge_key = edge_key.strip('()').replace(' ', '')
                parts = edge_key.split('-')
                if len(parts) == 2:
                    u, v = int(parts[0]), int(parts[1])
                else:
                    logger.warning(f"Invalid edge_key format: {edge_key}")
                    continue
            elif isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                u, v = int(u), int(v)
            else:
                logger.warning(f"Unknown edge_key type: {type(edge_key)}")
                continue

            # 边界检查
            if 0 <= u < n and 0 <= v < n:
                tree_mat[u, v] = float(flow)
            else:
                logger.warning(f"Invalid edge: ({u}, {v}), n={n}")

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse edge '{edge_key}': {e}")
            continue

    return tree_mat


def validate_hvt(hvt: np.ndarray, n: int, K_vnf: int) -> bool:
    """
    验证 HVT 矩阵的合法性

    Args:
        hvt: HVT 矩阵
        n: 节点数
        K_vnf: VNF 类型数

    Returns:
        是否合法
    """
    if hvt is None:
        logger.error("❌ hvt is None")
        return False

    if not isinstance(hvt, np.ndarray):
        logger.error(f"❌ hvt is not ndarray, got {type(hvt)}")
        return False

    if hvt.shape != (n, K_vnf):
        logger.error(f"❌ hvt shape mismatch: {hvt.shape} vs expected ({n}, {K_vnf})")
        return False

    if hvt.dtype != np.float32:
        logger.warning(f"⚠️ hvt dtype is {hvt.dtype}, expected float32")
        return False

    return True


def validate_tree(tree: Union[Dict, np.ndarray], n: int) -> bool:
    """
    验证树的合法性

    Args:
        tree: 树（字典或矩阵）
        n: 节点数

    Returns:
        是否合法
    """
    if tree is None:
        logger.error("❌ tree is None")
        return False

    if isinstance(tree, dict):
        # 字典形式的树
        for edge_key, flow in tree.items():
            if not isinstance(flow, (int, float)):
                logger.error(f"❌ Invalid flow type: {type(flow)}")
                return False
        return True

    elif isinstance(tree, np.ndarray):
        # 矩阵形式的树
        if tree.shape != (n, n):
            logger.error(f"❌ tree shape mismatch: {tree.shape} vs expected ({n}, {n})")
            return False
        return True

    else:
        logger.error(f"❌ tree type invalid: {type(tree)}")
        return False


def parse_vnf_key(vnf_key: Union[str, int]) -> tuple:
    """
    解析 VNF 键

    Args:
        vnf_key: "vnf_0_type_3" 或整数

    Returns:
        (vnf_idx, vnf_type) 或 (None, vnf_type)
    """
    if isinstance(vnf_key, str):
        parts = vnf_key.split('_')
        if len(parts) >= 4:
            try:
                vnf_idx = int(parts[1])
                vnf_type = int(parts[3])
                return vnf_idx, vnf_type
            except ValueError:
                pass
    elif isinstance(vnf_key, int):
        return None, vnf_key

    return None, None


def parse_edge_key(edge_key: Union[str, tuple]) -> tuple:
    """
    解析边键

    Args:
        edge_key: "0-1" 或 "(0, 1)" 或 (0, 1)

    Returns:
        (u, v) 或 (None, None)
    """
    try:
        if isinstance(edge_key, str):
            # 格式: "0-1" 或 "(0, 1)"
            edge_key = edge_key.strip('()').replace(' ', '')
            parts = edge_key.split('-')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        elif isinstance(edge_key, tuple) and len(edge_key) == 2:
            return int(edge_key[0]), int(edge_key[1])
    except (ValueError, IndexError):
        pass

    return None, None


def dict_to_hvt(placement_dict: Dict, n: int, K_vnf: int) -> np.ndarray:
    """
    字典转 HVT 矩阵（build_hvt_from_placement 的别名）

    为了兼容性保留
    """
    return build_hvt_from_placement(placement_dict, n, K_vnf)


def hvt_to_dict(hvt: np.ndarray) -> Dict:
    """
    HVT 矩阵转字典

    Args:
        hvt: shape (n, K_vnf) 的矩阵

    Returns:
        {vnf_key: node_idx} 字典
    """
    placement = {}

    indices = np.argwhere(hvt > 0)
    for idx, (node, vnf_type) in enumerate(indices):
        vnf_key = f"vnf_{idx}_type_{vnf_type}"
        placement[vnf_key] = int(node)

    return placement


def safe_dtype_convert(arr: Any, target_dtype=np.float32) -> np.ndarray:
    """
    安全的 dtype 转换

    Args:
        arr: 输入数组或可转换对象
        target_dtype: 目标类型

    Returns:
        转换后的数组
    """
    if arr is None:
        return None

    if isinstance(arr, dict):
        logger.warning("⚠️ Cannot convert dict to array directly")
        return None

    try:
        arr = np.asarray(arr, dtype=target_dtype)
        return arr
    except Exception as e:
        logger.error(f"❌ Failed to convert to {target_dtype}: {e}")
        return None


def merge_trees(tree1: Dict, tree2: Dict) -> Dict:
    """
    合并两个树（流量相加）

    Args:
        tree1: 树1
        tree2: 树2

    Returns:
        合并后的树
    """
    merged = dict(tree1)

    for edge_key, flow in tree2.items():
        if edge_key in merged:
            merged[edge_key] += flow
        else:
            merged[edge_key] = flow

    return merged


def get_vnf_count(hvt: np.ndarray) -> int:
    """
    获取 HVT 中部署的 VNF 总数

    Args:
        hvt: HVT 矩阵

    Returns:
        VNF 数量
    """
    return int(np.sum(hvt > 0))


def get_active_nodes(hvt: np.ndarray) -> List[int]:
    """
    获取部署了 VNF 的节点列表

    Args:
        hvt: HVT 矩阵

    Returns:
        节点索引列表
    """
    active_nodes = np.any(hvt > 0, axis=1)
    return np.where(active_nodes)[0].tolist()


# ============================================================================
# 导出所有函数
# ============================================================================
__all__ = [
    'ensure_list',
    'build_hvt_from_placement',
    'build_tree_vec',
    'validate_hvt',
    'validate_tree',
    'parse_vnf_key',
    'parse_edge_key',
    'dict_to_hvt',
    'hvt_to_dict',
    'safe_dtype_convert',
    'merge_trees',
    'get_vnf_count',
    'get_active_nodes',
]