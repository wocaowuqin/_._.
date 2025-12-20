# [file name]: diagnostic.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResourceManager 节点索引诊断工具
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
import copy

logger = logging.getLogger(__name__)


class ResourceManagerDiagnostic:
    """ResourceManager 诊断工具"""

    def __init__(self, resource_manager):
        self.rm = resource_manager
        self.diagnostic_results = {}

    def run_full_diagnostic(self):
        """运行完整诊断"""
        print("\n" + "=" * 80)
        print("ResourceManager 节点索引诊断报告")
        print("=" * 80)

        self.diagnostic_results = {
            'basic_info': self._check_basic_info(),
            'node_index_base': self._check_node_index_base(),
            'dc_nodes_conversion': self._check_dc_nodes_conversion(),
            'link_map_consistency': self._check_link_map_consistency(),
            'normalize_node_function': self._test_normalize_node(),
            'request_conversion': self._test_request_conversion(),
            'feasibility_check': self._test_feasibility_check(),
            'resource_access': self._test_resource_access(),
            'summary': {}
        }

        # 生成总结
        self._generate_summary()

        return self.diagnostic_results

    def _check_basic_info(self):
        """检查基础信息"""
        info = {
            'init_mode': self.rm.init_mode,
            'node_count': self.rm.n,
            'link_count': self.rm.L,
            'vnf_types': self.rm.K_vnf,
            'capacities': {
                'cpu_cap': self.rm.C_cap,
                'mem_cap': self.rm.M_cap,
                'bw_cap': self.rm.B_cap
            }
        }

        print(f"\n[1] 基础信息:")
        print(f"   初始化模式: {info['init_mode']}")
        print(f"   节点数: {info['node_count']} (0-based索引范围: 0-{info['node_count'] - 1})")
        print(f"   链路数: {info['link_count']}")

        return info

    def _check_node_index_base(self):
        """检查节点索引基值"""
        base = self.rm.node_index_base

        print(f"\n[2] 节点索引基值:")
        print(f"   node_index_base = {base}")
        print(f"   → 外部节点ID: {1 if base == 1 else 0}-based")
        print(f"   → 内部节点ID: 0-based")

        return {
            'node_index_base': base,
            'external_base': 1 if base == 1 else 0,
            'internal_base': 0
        }

    def _check_dc_nodes_conversion(self):
        """检查DC节点转换"""
        print(f"\n[3] DC节点转换检查:")

        # 检查初始化的dc_nodes列表
        dc_nodes_raw = self.rm.dc_nodes if hasattr(self.rm, 'dc_nodes') else []

        # 验证所有DC节点都在有效范围内
        valid_nodes = []
        invalid_nodes = []
        out_of_range_nodes = []

        for node in dc_nodes_raw:
            if isinstance(node, (int, np.integer)):
                if 0 <= node < self.rm.n:
                    valid_nodes.append(node)
                else:
                    out_of_range_nodes.append(node)
            else:
                invalid_nodes.append(node)

        print(f"   DC节点列表: {sorted(dc_nodes_raw)}")
        print(f"   有效DC节点: {sorted(valid_nodes)} (共{len(valid_nodes)}个)")

        if out_of_range_nodes:
            print(f"   ⚠️  超出范围的DC节点: {out_of_range_nodes}")

        if invalid_nodes:
            print(f"   ❌ 无效的DC节点类型: {invalid_nodes}")

        # 检查在特征构建中是否正确使用
        if hasattr(self.rm, '_build_node_features'):
            # 抽样检查几个节点
            test_nodes = [0, min(5, self.rm.n - 1), self.rm.n // 2, self.rm.n - 1]
            dc_check_results = []

            for node_idx in test_nodes:
                is_dc = node_idx in dc_nodes_raw
                dc_check_results.append((node_idx, is_dc))

            print(f"   特征构建检查: 节点{test_nodes}是否标记为DC → {dc_check_results}")

        return {
            'dc_nodes_raw': sorted(dc_nodes_raw),
            'valid_dc_nodes': sorted(valid_nodes),
            'out_of_range_dc_nodes': out_of_range_nodes,
            'invalid_dc_nodes': invalid_nodes
        }

    def _check_link_map_consistency(self):
        """检查链路映射一致性"""
        print(f"\n[4] 链路映射检查:")

        if not hasattr(self.rm, 'link_map') or not self.rm.link_map:
            print("   ⚠️  无链路映射信息")
            return {'has_link_map': False}

        link_map = self.rm.link_map
        total_edges = len(link_map) // 2  # 双向边算一条

        # 检查链路键的格式
        sample_edges = list(link_map.keys())[:5]
        edge_types = {}

        for edge in sample_edges:
            u, v = edge
            key_type = f"({type(u).__name__}, {type(v).__name__})"
            edge_types[key_type] = edge_types.get(key_type, 0) + 1

        print(f"   链路总数: {total_edges}条物理链路")
        print(f"   链路ID范围: {min(link_map.values())} - {max(link_map.values())}")
        print(f"   样本边格式: {edge_types}")

        # 检查边索引是否一致
        consistency_issues = []
        for (u, v), lid in list(link_map.items())[:10]:
            # 检查反向边是否也存在且ID相同
            reverse_key = (v, u)
            if reverse_key in link_map:
                if link_map[reverse_key] != lid:
                    consistency_issues.append(f"({u},{v}):{lid} vs ({v},{u}):{link_map[reverse_key]}")
            else:
                consistency_issues.append(f"缺少反向边: ({u},{v})")

        if consistency_issues:
            print(f"   ⚠️  链路一致性问题: {consistency_issues[:3]}")
        else:
            print(f"   ✅ 链路映射一致性良好")

        return {
            'has_link_map': True,
            'total_physical_links': total_edges,
            'link_id_range': (min(link_map.values()), max(link_map.values())),
            'edge_types': edge_types,
            'consistency_issues': consistency_issues[:5]
        }

    def _test_normalize_node(self):
        """测试_normalize_node函数"""
        print(f"\n[5] 节点ID转换函数测试:")

        test_cases = []

        # 测试典型节点
        test_nodes = [0, 1, 10, self.rm.n - 1]

        for external_node in test_nodes:
            try:
                internal_node = self.rm._normalize_node(external_node, from_external=True)
                reverse_check = self.rm._normalize_node(internal_node, from_external=False)

                test_cases.append({
                    'external': external_node,
                    'internal': internal_node,
                    'reverse': reverse_check,
                    'consistent': external_node == reverse_check
                })
            except Exception as e:
                test_cases.append({
                    'external': external_node,
                    'error': str(e)
                })

        for tc in test_cases:
            if 'error' in tc:
                print(f"   外部节点{tc['external']} → 错误: {tc['error']}")
            else:
                status = "✅" if tc['consistent'] else "❌"
                print(f"   {status} 外部{tc['external']} → 内部{tc['internal']} → 外部{tc['reverse']}")

        return test_cases

    def _test_request_conversion(self):
        """测试请求转换"""
        print(f"\n[6] 请求转换测试:")

        # 创建测试请求
        test_requests = [
            {
                'id': 1,
                'source': 1,  # 1-based
                'dest': [3, 5, 7],  # 1-based
                'vnf': [1, 2, 3],
                'bw_origin': 7,
                'cpu_origin': [10, 8, 6],
                'memory_origin': [8, 6, 4]
            },
            {
                'id': 2,
                'source': 0,  # 0-based
                'destinations': [2, 4, 6],  # 0-based
                'vnf': [2, 3, 1],
                'bw_origin': 5,
                'cpu_origin': [8, 7, 5],
                'memory_origin': [6, 5, 3]
            }
        ]

        conversion_results = []

        for i, req in enumerate(test_requests):
            print(f"\n   测试请求 {i + 1}:")
            print(f"     原始: source={req.get('source')}, dests={req.get('dest') or req.get('destinations')}")

            # 转换source
            try:
                source_external = req.get('source')
                source_internal = self.rm._normalize_node(source_external, from_external=True)

                # 转换destinations
                dests_field = req.get('destinations') or req.get('dest')
                dests_internal = [self.rm._normalize_node(d, from_external=True) for d in dests_field]

                # 边界检查
                source_valid = 0 <= source_internal < self.rm.n
                dests_valid = all(0 <= d < self.rm.n for d in dests_internal)

                print(f"     转换后: source={source_internal}({'有效' if source_valid else '无效'}), "
                      f"dests={dests_internal}({'全部有效' if dests_valid else '有无效节点'})")

                conversion_results.append({
                    'request_id': i,
                    'source_external': source_external,
                    'source_internal': source_internal,
                    'source_valid': source_valid,
                    'dests_external': dests_field,
                    'dests_internal': dests_internal,
                    'dests_valid': dests_valid
                })

            except Exception as e:
                print(f"     ❌ 转换失败: {e}")
                conversion_results.append({
                    'request_id': i,
                    'error': str(e)
                })

        return conversion_results

    def _test_feasibility_check(self):
        """测试可行性检查"""
        print(f"\n[7] 全局可行性检查测试:")

        if not hasattr(self.rm, 'check_global_feasibility'):
            print("   ⚠️  check_global_feasibility方法不存在")
            return {'method_exists': False}

        # 创建测试状态
        test_state = self.rm.create_initial_state()

        # 测试请求（基于当前系统的索引基值）
        base = self.rm.node_index_base

        if base == 1:
            # 1-based 系统
            test_req = {
                'id': 99,
                'source': 1 if 1 in self.rm.dc_nodes else min(self.rm.dc_nodes) + 1,
                'dest': [min(3, self.rm.n), min(5, self.rm.n), min(7, self.rm.n)],
                'vnf': [1, 2, 3],
                'bw_origin': 5,
                'cpu_origin': [8, 6, 4],
                'memory_origin': [6, 4, 3]
            }
        else:
            # 0-based 系统
            test_req = {
                'id': 99,
                'source': 0 if 0 in self.rm.dc_nodes else min(self.rm.dc_nodes),
                'destinations': [2, 4, 6],
                'vnf': [1, 2, 3],
                'bw_origin': 5,
                'cpu_origin': [8, 6, 4],
                'memory_origin': [6, 4, 3]
            }

        print(f"   测试请求: source={test_req.get('source')}, "
              f"dests={test_req.get('dest') or test_req.get('destinations')}")

        try:
            result = self.rm.check_global_feasibility(test_req, test_state)
            print(f"   可行性检查结果: {result}")

            return {
                'method_exists': True,
                'test_request': test_req,
                'feasibility_result': result,
                'state_used': bool(test_state)
            }
        except Exception as e:
            print(f"   ❌ 可行性检查失败: {e}")
            return {
                'method_exists': True,
                'error': str(e)
            }

    def _test_resource_access(self):
        """测试资源访问"""
        print(f"\n[8] 资源访问测试:")

        # 测试几个节点的资源检查
        test_nodes = [0, min(3, self.rm.n - 1), self.rm.n // 2, min(self.rm.n - 1, 10)]

        access_results = []

        for node in test_nodes:
            try:
                # 测试节点资源检查
                cpu_check = self.rm.check_node_resources(node, 10, 8, external_index=False)

                # 测试资源更新
                update_success = self.rm.update_resources(node, -5, -4, strict=False, external_index=False)

                access_results.append({
                    'node': node,
                    'cpu_check': cpu_check,
                    'update_success': update_success,
                    'current_cpu': self.rm.C[node] if node < len(self.rm.C) else None,
                    'current_mem': self.rm.M[node] if node < len(self.rm.M) else None
                })

                print(f"   节点{node}: 资源检查={cpu_check}, 更新成功={update_success}, "
                      f"当前(CPU={self.rm.C[node]:.1f}, MEM={self.rm.M[node]:.1f})")

            except Exception as e:
                print(f"   节点{node}: ❌ 访问失败 - {e}")
                access_results.append({
                    'node': node,
                    'error': str(e)
                })

        return access_results

    def _generate_summary(self):
        """生成诊断总结"""
        print(f"\n" + "=" * 80)
        print("诊断总结")
        print("=" * 80)

        issues = []
        warnings = []

        # 检查DC节点有效性
        dc_info = self.diagnostic_results.get('dc_nodes_conversion', {})
        if dc_info.get('out_of_range_dc_nodes') or dc_info.get('invalid_dc_nodes'):
            issues.append("DC节点包含无效或超出范围的节点")

        # 检查节点转换一致性
        norm_test = self.diagnostic_results.get('normalize_node_function', [])
        for test in norm_test:
            if 'error' in test:
                issues.append(f"节点转换函数错误: {test['error']}")
            elif not test.get('consistent', False):
                issues.append(f"节点转换不一致: 外部{test['external']}≠{test['reverse']}")

        # 检查请求转换
        req_test = self.diagnostic_results.get('request_conversion', [])
        for test in req_test:
            if 'error' in test:
                issues.append(f"请求转换错误: {test['error']}")
            elif not test.get('source_valid', False) or not test.get('dests_valid', False):
                warnings.append(f"请求包含无效节点")

        # 检查可行性检查
        feas_test = self.diagnostic_results.get('feasibility_check', {})
        if 'error' in feas_test:
            issues.append(f"可行性检查错误: {feas_test['error']}")

        # 检查资源访问
        res_test = self.diagnostic_results.get('resource_access', [])
        for test in res_test:
            if 'error' in test:
                issues.append(f"资源访问错误: {test['error']}")

        # 输出总结
        if issues:
            print("❌ 发现严重问题:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ 未发现严重问题")

        if warnings:
            print("\n⚠️  警告:")
            for warning in warnings:
                print(f"   - {warning}")

        # 索引一致性建议
        base = self.rm.node_index_base
        print(f"\n📋 索引一致性建议:")
        print(f"   当前系统使用 {base}-based 外部索引")
        print(f"   确保以下一致性:")
        print(f"   1. 所有外部输入（请求、配置）使用 {base}-based 索引")
        print(f"   2. ResourceManager 内部使用 0-based 索引")
        print(f"   3. 链路映射键使用 0-based 索引")
        print(f"   4. DC节点列表使用 0-based 索引")

        self.diagnostic_results['summary'] = {
            'issues': issues,
            'warnings': warnings,
            'external_index_base': base,
            'internal_index_base': 0
        }


# 使用示例
def diagnose_resource_manager(resource_manager):
    """运行ResourceManager诊断"""
    diagnostic = ResourceManagerDiagnostic(resource_manager)
    return diagnostic.run_full_diagnostic()


# 集成到ResourceManager中的快速诊断方法
def add_diagnostic_to_resource_manager():
    """为ResourceManager添加快速诊断方法"""

    def quick_diagnose(self):
        """快速诊断ResourceManager状态"""
        print("\n" + "=" * 60)
        print(f"ResourceManager 快速诊断 (节点数: {self.n})")
        print("=" * 60)

        # 1. 检查索引基值
        print(f"1. 索引基值: node_index_base = {self.node_index_base}")
        print(f"   → 外部: {1 if self.node_index_base == 1 else 0}-based")
        print(f"   → 内部: 0-based")

        # 2. 检查DC节点
        print(f"2. DC节点: {sorted(self.dc_nodes)}")
        print(f"   有效范围: 0-{self.n - 1}")

        invalid_dc = [n for n in self.dc_nodes if not (0 <= n < self.n)]
        if invalid_dc:
            print(f"   ⚠️  无效DC节点: {invalid_dc}")

        # 3. 检查链路映射
        if hasattr(self, 'link_map') and self.link_map:
            sample_edge = list(self.link_map.keys())[0]
            u, v = sample_edge
            print(f"3. 链路映射样本: ({u},{v}) -> ID {self.link_map[sample_edge]}")
            print(f"   键类型: ({type(u).__name__}, {type(v).__name__})")

        # 4. 测试节点转换
        print("4. 节点转换测试:")
        test_nodes = [0, 1, self.n - 1] if self.n > 1 else [0]
        for node in test_nodes:
            internal = self._normalize_node(node, from_external=True)
            external = self._normalize_node(internal, from_external=False)
            print(f"   外部{node} → 内部{internal} → 外部{external} "
                  f"{'✅' if node == external else '❌'}")

        # 5. 资源状态
        print("5. 资源状态样本:")
        for i in range(min(3, self.n)):
            print(f"   节点{i}: CPU={self.C[i]:.1f}/{self.C_cap}, "
                  f"MEM={self.M[i]:.1f}/{self.M_cap}")

        print("=" * 60)
        return True

    # 添加到ResourceManager类
    from resource_manager import ResourceManager
    ResourceManager.quick_diagnose = quick_diagnose