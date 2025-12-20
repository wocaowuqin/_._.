#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源耗尽部署诊断工具 (自适应版)
修复: 自动适配 ResourceManager 缺失的方法，定位 0 失败的根源
"""

import numpy as np
import logging
import sys
import os

# 路径适配
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResourceExhaustionDiagnostics:
    def __init__(self):
        self.issues = []

    def print_header(self, title):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_section(self, title):
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")

    def test_zero_capacity_initialization(self):
        """测试 1: 零容量配置检测"""
        self.print_header("测试 1: 零容量配置检测")
        try:
            from utils.config_utils import load_config
            config = load_config('phase1')
            capacities = config.get('capacities', {})

            # 如果配置里没写，检查默认值
            if not capacities:
                # 尝试从 env.capacities 获取
                env_conf = config.get('env', {}).get('capacities', {})
                if env_conf: capacities = env_conf

            print(f"\n📋 配置文件中的容量:")
            print(f"   CPU:       {capacities.get('cpu', 'N/A')}")
            print(f"   Memory:    {capacities.get('memory', 'N/A')}")
            print(f"   Bandwidth: {capacities.get('bandwidth', 'N/A')}")

            issues = []
            if float(capacities.get('cpu', 0)) <= 0: issues.append("CPU 容量 <= 0")
            if float(capacities.get('memory', 0)) <= 0: issues.append("Memory 容量 <= 0")

            if issues:
                print(f"\n❌ 发现问题:")
                for issue in issues: print(f"   - {issue}")
                self.issues.append(("配置错误", str(issues)))
            else:
                print(f"\n✅ 容量配置看起来正常")
            return config, capacities
        except Exception as e:
            logger.error(f"配置检测失败: {e}")
            return None, {}

    def test_resource_behavior(self, config, capacities):
        """测试 2: 资源管理器的实际行为"""
        self.print_header("测试 2: 资源管理器行为分析")

        if config is None: return

        try:
            # 1. 初始化 ResourceManager
            # 尝试从 envs.modules.resource 导入
            try:
                from envs.modules.resource import ResourceManager
            except ImportError:
                print("❌ 无法导入 ResourceManager，请检查路径")
                return

            # 模拟拓扑数据
            topo = np.ones((28, 28), dtype=np.float32)
            np.fill_diagonal(topo, 0)
            dc_nodes = list(range(10))

            # 实例化
            rm = ResourceManager(topo, capacities, dc_nodes)
            print(f"\n✅ ResourceManager 初始化成功")

            # 2. 分析内部数据结构
            print(f"\n🔍 内部数据结构探测:")
            has_check_method = hasattr(rm, 'check_node_resources')
            has_update_method = hasattr(rm, 'update_resources')

            print(f"   - check_node_resources 方法: {'存在' if has_check_method else '❌ 不存在 (这就是原因!)'}")
            print(f"   - update_resources 方法:     {'存在' if has_update_method else '❌ 不存在'}")

            # 检查资源数组
            cpu_arr = None
            if hasattr(rm, 'C'):
                cpu_arr = rm.C
            elif hasattr(rm, 'cpu_state'):
                cpu_arr = rm.cpu_state

            if cpu_arr is not None:
                print(f"   - CPU 数组值 (前5个): {cpu_arr[:5]}")
                if np.sum(cpu_arr) == 0:
                    print(f"   ❌ 严重: CPU 数组全为 0 (初始化失败)")
                    self.issues.append(("初始化", "CPU数组全为0"))
            else:
                print(f"   ❌ 无法找到 CPU 存储数组 (C 或 cpu_state)")
                return

            # 3. 模拟资源检查 (如果没有方法，我们手动检查逻辑)
            self.print_section("资源检查逻辑测试")

            node_idx = 0
            req_cpu = 99999.0  # 超大请求

            print(f"测试场景: 节点 {node_idx} (剩余 {cpu_arr[node_idx]}) 请求 {req_cpu} CPU")

            if has_check_method:
                res = rm.check_node_resources(node_idx, req_cpu, 0)
                print(f"   👉 调用类方法检查结果: {res}")
                if res:
                    print("   ❌ 错误: 资源不足却返回 True")
                    self.issues.append(("逻辑错误", "检查方法失效"))
            else:
                print("   ⚠️  类没有检查方法，完全依赖外部 (Env/Agent) 检查。")
                print("   👉 如果外部没写 `if available < needed: return False`，就会导致 0 失败。")

            # 4. 模拟扣除与负数测试
            self.print_section("负资源容忍度测试")

            print(f"尝试强制扣除，看是否会变成负数...")
            original_val = cpu_arr[node_idx]

            try:
                # 尝试各种扣除方式
                if has_update_method:
                    rm.update_resources(node_idx, -req_cpu, 0)  # 假设负数是扣除
                elif hasattr(rm, 'consume_cpu'):
                    rm.consume_cpu(node_idx, req_cpu)
                else:
                    # 暴力修改
                    cpu_arr[node_idx] -= req_cpu

                new_val = cpu_arr[node_idx]
                print(f"   扣除后 CPU: {new_val}")

                if new_val < 0:
                    print(f"   ❌ 严重: 资源变成了负数 ({new_val})！")
                    print(f"   ✅ 结论: 系统允许透支，所以永远不会失败。")
                    self.issues.append(("致命逻辑", "资源允许变成负数"))
                else:
                    print(f"   ✅ 资源未变负 (可能有保护逻辑)")

            except Exception as e:
                print(f"   执行扣除时报错: {e}")

        except Exception as e:
            logger.error(f"测试中断: {e}")
            import traceback
            traceback.print_exc()

    def print_report(self):
        self.print_header("最终诊断结论")
        if not self.issues:
            print("✅ 未发现明显逻辑漏洞。")
        else:
            for cat, msg in self.issues:
                print(f"❌ [{cat}] {msg}")

            print("\n💡 修复建议:")
            if any("不存在" in str(msg) for _, msg in self.issues):
                print("1. 你的 ResourceManager 缺少检查逻辑。")
                print("   在 envs/sfc_env.py 的部署逻辑中，必须手动添加：")
                print("   if node_cpu_rem < req_cpu: return False")

            if any("变成负数" in str(msg) for _, msg in self.issues):
                print("2. 系统没有防止资源透支。")
                print("   在 update_resources 中添加：assert new_val >= 0")


if __name__ == '__main__':
    diag = ResourceExhaustionDiagnostics()
    config, capacities = diag.test_zero_capacity_initialization()
    diag.test_resource_behavior(config, capacities)
    diag.print_report()