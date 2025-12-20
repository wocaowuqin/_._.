#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Mechanics & Resource Check (Enhanced)
增强版：支持多种 ResourceManager 实现
"""

import os
import sys
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger("EnvTest")


def get_total_resources(env):
    """
    获取全网剩余资源总和
    支持多种 ResourceManager 实现
    """
    rm = env.resource_mgr
    total_cpu = 0.0
    total_mem = 0.0
    total_bw = 0.0

    # ========================================
    # 方法 1: 直接从数组获取（最常见）
    # ========================================
    if hasattr(rm, 'C') and rm.C is not None:
        total_cpu = float(np.sum(rm.C))
        logger.debug(f"  [资源检测] 从 rm.C 获取 CPU: {total_cpu:.2f}")

    if hasattr(rm, 'M') and rm.M is not None:
        total_mem = float(np.sum(rm.M))
        logger.debug(f"  [资源检测] 从 rm.M 获取 MEM: {total_mem:.2f}")

    if hasattr(rm, 'B') and rm.B is not None:
        total_bw = float(np.sum(rm.B))
        logger.debug(f"  [资源检测] 从 rm.B 获取 BW: {total_bw:.2f}")

    # ========================================
    # 方法 2: 从 nodes/links 字典获取
    # ========================================
    if total_cpu == 0 and hasattr(rm, 'nodes'):
        if isinstance(rm.nodes, dict):
            # nodes 是字典 {'cpu': array, 'memory': array}
            if 'cpu' in rm.nodes:
                total_cpu = float(np.sum(rm.nodes['cpu']))
                logger.debug(f"  [资源检测] 从 rm.nodes['cpu'] 获取: {total_cpu:.2f}")
            if 'memory' in rm.nodes:
                total_mem = float(np.sum(rm.nodes['memory']))
        elif isinstance(rm.nodes, (list, np.ndarray)):
            # nodes 是列表
            for node in rm.nodes:
                if isinstance(node, dict):
                    total_cpu += node.get('cpu', 0)
                    total_mem += node.get('memory', 0)

    if total_bw == 0 and hasattr(rm, 'links'):
        if isinstance(rm.links, dict) and 'bandwidth' in rm.links:
            # links 是字典 {'bandwidth': {(u,v): bw}}
            bw_dict = rm.links['bandwidth']
            total_bw = float(sum(bw_dict.values()))
            logger.debug(f"  [资源检测] 从 rm.links['bandwidth'] 获取: {total_bw:.2f}")

    # ========================================
    # 方法 3: 备用属性名
    # ========================================
    if total_cpu == 0 and hasattr(rm, 'node_cap'):
        total_cpu = float(np.sum(rm.node_cap))
        logger.debug(f"  [资源检测] 从 rm.node_cap 获取: {total_cpu:.2f}")

    if total_mem == 0 and hasattr(rm, 'node_mem'):
        total_mem = float(np.sum(rm.node_mem))
        logger.debug(f"  [资源检测] 从 rm.node_mem 获取: {total_mem:.2f}")

    if total_bw == 0 and hasattr(rm, 'link_cap'):
        total_bw = float(np.sum(rm.link_cap))
        logger.debug(f"  [资源检测] 从 rm.link_cap 获取: {total_bw:.2f}")

    return total_cpu, total_mem, total_bw


def diagnose_resource_manager(env):
    """诊断 ResourceManager 的结构"""
    rm = env.resource_mgr

    logger.info("\n" + "=" * 70)
    logger.info("🔍 ResourceManager 结构诊断")
    logger.info("=" * 70)

    # 检查类名
    logger.info(f"类名: {rm.__class__.__name__}")
    logger.info(f"模块: {rm.__class__.__module__}")

    # 检查基本属性
    attrs_to_check = [
        'n', 'node_num', 'L', 'link_num', 'K_vnf', 'type_num',
        'C', 'M', 'B',
        'C_cap', 'M_cap', 'B_cap',
        'cap_cpu', 'cap_mem', 'cap_bw',
        'nodes', 'links',
        'node_cap', 'node_mem', 'link_cap'
    ]

    logger.info("\n属性检查:")
    for attr in attrs_to_check:
        if hasattr(rm, attr):
            val = getattr(rm, attr)
            if isinstance(val, np.ndarray):
                logger.info(f"  ✅ {attr}: array shape={val.shape}, sum={np.sum(val):.2f}")
            elif isinstance(val, dict):
                logger.info(f"  ✅ {attr}: dict with {len(val)} keys")
            elif isinstance(val, (int, float)):
                logger.info(f"  ✅ {attr}: {val}")
            else:
                logger.info(f"  ✅ {attr}: {type(val)}")
        else:
            logger.info(f"  ❌ {attr}: NOT FOUND")

    # 检查容量配置
    logger.info("\n容量配置:")
    cap_attrs = ['C_cap', 'M_cap', 'B_cap', 'cap_cpu', 'cap_mem', 'cap_bw']
    for attr in cap_attrs:
        if hasattr(rm, attr):
            logger.info(f"  {attr} = {getattr(rm, attr)}")


def test_resource_initialization(env):
    """测试资源初始化"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 [测试 1] 资源初始化检查")
    logger.info("=" * 70)

    env.reset()
    total_cpu, total_mem, total_bw = get_total_resources(env)

    logger.info(f"\n全网资源统计:")
    logger.info(f"  总 CPU: {total_cpu:.2f}")
    logger.info(f"  总 Memory: {total_mem:.2f}")
    logger.info(f"  总 Bandwidth: {total_bw:.2f}")

    # 验证
    passed = True

    if total_cpu <= 0:
        logger.error("❌ CPU 资源为 0 或负数")
        passed = False
    else:
        logger.info(f"✅ CPU 资源正常")

    if total_mem <= 0:
        logger.error("❌ Memory 资源为 0 或负数")
        passed = False
    else:
        logger.info(f"✅ Memory 资源正常")

    if total_bw <= 0:
        logger.warning("⚠️  Bandwidth 资源为 0（可能正常，取决于拓扑）")
    else:
        logger.info(f"✅ Bandwidth 资源正常")

    return passed


def test_resource_consumption(env):
    """测试资源消耗"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 [测试 2] 资源消耗测试")
    logger.info("=" * 70)

    rm = env.resource_mgr

    # 记录初始资源
    cpu_before, mem_before, bw_before = get_total_resources(env)
    logger.info(f"\n初始资源:")
    logger.info(f"  CPU: {cpu_before:.2f}")
    logger.info(f"  Memory: {mem_before:.2f}")
    logger.info(f"  Bandwidth: {bw_before:.2f}")

    # 尝试手动消耗资源
    logger.info(f"\n尝试消耗资源...")

    consumed = False

    # 方法 1: 直接修改数组
    if hasattr(rm, 'C') and rm.C is not None:
        node_idx = 0
        cpu_cost = 10.0
        if rm.C[node_idx] >= cpu_cost:
            rm.C[node_idx] -= cpu_cost
            logger.info(f"  ✅ 从 rm.C[{node_idx}] 扣除 {cpu_cost} CPU")
            consumed = True

    # 方法 2: 修改 nodes 字典
    elif hasattr(rm, 'nodes') and isinstance(rm.nodes, dict) and 'cpu' in rm.nodes:
        node_idx = 0
        cpu_cost = 10.0
        if rm.nodes['cpu'][node_idx] >= cpu_cost:
            rm.nodes['cpu'][node_idx] -= cpu_cost
            logger.info(f"  ✅ 从 rm.nodes['cpu'][{node_idx}] 扣除 {cpu_cost} CPU")
            consumed = True

    if not consumed:
        logger.warning("  ⚠️  无法找到可修改的资源属性")
        return False

    # 检查资源是否减少
    cpu_after, mem_after, bw_after = get_total_resources(env)
    logger.info(f"\n消耗后资源:")
    logger.info(f"  CPU: {cpu_after:.2f} (减少 {cpu_before - cpu_after:.2f})")
    logger.info(f"  Memory: {mem_after:.2f}")
    logger.info(f"  Bandwidth: {bw_after:.2f}")

    if cpu_after < cpu_before:
        logger.info("✅ 资源消耗成功！")
        return True
    else:
        logger.error("❌ 资源未减少，消耗失败！")
        return False


def main():
    logger.info("🚀 启动增强版环境资源检测...")

    # 1. 加载配置
    config = load_config('phase1')

    # 2. 初始化环境
    logger.info("\n正在初始化环境...")
    try:
        env = SFC_HIRL_Env(config)
    except Exception as e:
        logger.error(f"❌ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("✅ 环境初始化成功")

    # 3. 诊断 ResourceManager
    diagnose_resource_manager(env)

    # 4. 测试资源初始化
    test1_passed = test_resource_initialization(env)

    # 5. 测试资源消耗
    test2_passed = test_resource_consumption(env)

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("📊 测试总结")
    logger.info("=" * 70)
    logger.info(f"资源初始化: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    logger.info(f"资源消耗:   {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if not test1_passed:
        logger.error("\n❌ 资源初始化失败！")
        logger.error("   可能原因:")
        logger.error("   1. ResourceManager.__init__ 中没有正确初始化 C, M, B 数组")
        logger.error("   2. capacities 配置为空或 0")
        logger.error("   3. nodes/links 字典未创建")
        logger.info("\n   修复建议:")
        logger.info("   在 envs/modules/resource.py 的 __init__ 中添加:")
        logger.info("   ```python")
        logger.info("   self.C = np.full(self.n, self.C_cap, dtype=float)")
        logger.info("   self.M = np.full(self.n, self.M_cap, dtype=float)")
        logger.info("   self.nodes = {'cpu': self.C.copy(), 'memory': self.M.copy()}")
        logger.info("   ```")


if __name__ == "__main__":
    main()