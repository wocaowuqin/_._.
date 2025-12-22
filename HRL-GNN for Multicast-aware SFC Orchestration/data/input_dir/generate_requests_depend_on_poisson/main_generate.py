#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的数据生成配置

关键修改：
1. 源节点从 DC 节点中选择（而不是非DC节点）
2. 目标节点可以是任意节点
"""

import random
import numpy as np
import pickle

# ========== 网络配置 ==========
NUM_NODES = 28

# ✅ DC 节点（数据中心，可以部署VNF）
#DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]
DC_NODES = [11]

# ✅ 重要节点（汇聚节点，可以作为目标，但不能作为源）
#NODE_IMPORTANT = [16, 21, 22, 24, 25, 26, 27, 28]
NODE_IMPORTANT = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20, 21, 22,23, 24, 25, 26, 27, 28]
# ========== 请求参数 ==========
NUM_DESTINATIONS = 5  # 每个请求的目标数
VNF_CHAIN_LENGTH = 3  # VNF链长度
VNF_TYPES = 8  # VNF类型数

# 带宽范围
MIN_BANDWIDTH = 4
MAX_BANDWIDTH = 8

# 持续时间参数
MEAN_LIFETIME = 3

# 泊松到达率
LAMBDA_RATE = 0.6

# 时间范围
T = 200  # Phase 1: 200个时间单位


def generate_single_request(req_id, source, destinations, vnf_chain, bandwidth,
                            cpu_needs, mem_needs, arrive_time, lifetime):
    """
    生成单个请求

    Args:
        req_id: 请求ID
        source: 源节点（必须是DC节点）
        destinations: 目标节点列表
        vnf_chain: VNF类型列表
        bandwidth: 带宽需求
        cpu_needs: CPU需求列表
        mem_needs: 内存需求列表
        arrive_time: 到达时间
        lifetime: 持续时间
    """
    leave_time = arrive_time + lifetime
    # ✅ 添加时间步字段
    arrive_time_step = int(np.ceil(arrive_time))
    leave_time_step = int(np.ceil(leave_time))
    return {
        'id': req_id,
        'source': source,
        'dest': destinations,
        'vnf': vnf_chain,
        'bw_origin': bandwidth,
        'cpu_origin': cpu_needs,
        'memory_origin': mem_needs,
        'arrival_time': arrive_time,
        'leave_time': leave_time,
        'lifetime': lifetime,
        'arrive_time_step': arrive_time_step,  # ✅ 新增
        'leave_time_step': leave_time_step
    }


def generate_poisson_arrivals(T, lamda):
    """生成泊松到达时间序列"""
    arrivals = []
    time_state = 0

    while time_state < T:
        # 指数分布的时间间隔
        interval = np.random.exponential(1.0 / lamda)
        time_state += interval

        if time_state < T:
            arrivals.append(time_state)

    return arrivals


def generate_vnf_resources(vnf_type, bandwidth):
    """
    根据VNF类型和带宽生成资源需求

    与MATLAB版本一致：
    - cpu_need = bandwidth * (rand * 2.75 + 0.25)
    - memory_need = bandwidth * (rand * 1.75 + 0.25)
    """
    cpu_factor = np.random.rand() * 2.75 + 0.25
    mem_factor = np.random.rand() * 1.75 + 0.25

    cpu = round(bandwidth * cpu_factor)
    mem = round(bandwidth * mem_factor)

    return cpu, mem


def generate_requests_for_source(source_node, arrive_times, all_nodes):
    """
    为单个源节点生成请求序列

    Args:
        source_node: 源节点（DC节点）
        arrive_times: 到达时间列表
        all_nodes: 所有可用节点（用于选择目标）
    """
    requests = []

    # 可选的目标节点（除了源节点自己）
    available_dests = [n for n in all_nodes if n != source_node]

    for i, arrive_time in enumerate(arrive_times):
        # 随机选择目标节点
        destinations = random.sample(available_dests, NUM_DESTINATIONS)

        # 随机选择VNF链
        vnf_chain = random.sample(range(1, VNF_TYPES + 1), VNF_CHAIN_LENGTH)

        # 随机带宽
        bandwidth = random.randint(MIN_BANDWIDTH, MAX_BANDWIDTH)

        # 计算资源需求
        cpu_needs = []
        mem_needs = []
        for vnf_type in vnf_chain:
            cpu, mem = generate_vnf_resources(vnf_type, bandwidth)
            cpu_needs.append(cpu)
            mem_needs.append(mem)

        # 生成持续时间（负指数分布，确保 1 < lifetime < 6）
        lifetime = 1 + np.random.exponential(MEAN_LIFETIME - 1)
        while lifetime > 6:
            lifetime = 1 + np.random.exponential(MEAN_LIFETIME - 1)

        # 创建请求
        req = generate_single_request(
            req_id=i + 1,  # 临时ID，后续会重新编号
            source=source_node,
            destinations=destinations,
            vnf_chain=vnf_chain,
            bandwidth=bandwidth,
            cpu_needs=cpu_needs,
            mem_needs=mem_needs,
            arrive_time=arrive_time,
            lifetime=lifetime
        )

        requests.append(req)

    return requests


def generate_all_requests(T=200, lamda=0.6, seed=None):
    """
    生成所有请求 (完全复刻 MATLAB 逻辑)

    修改点：
    1. 源节点遍历 NODE_IMPORTANT
    2. 目的节点池限制为 NODE_IMPORTANT
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("=" * 70)
    print("生成请求数据 (MATLAB Logic Mode)")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  时间范围: {T}")
    print(f"  到达率: {lamda}")
    print(f"  重要节点集合 (Source & Dest Pool): {NODE_IMPORTANT}")
    print(f"  VNF链长度: {VNF_CHAIN_LENGTH}")
    print(f"  目标数量: {NUM_DESTINATIONS}")

    all_requests = []

    # ✅ 关键修改 1：遍历重要节点 (MATLAB: for s = 1 : length(node_important))
    for source_node in NODE_IMPORTANT:
        print(f"\n为重要节点 {source_node} 生成请求...")

        # 生成到达时间
        arrive_times = generate_poisson_arrivals(T, lamda)
        print(f"  生成了 {len(arrive_times)} 个到达事件")

        # ✅ 关键修改 2：目标节点池限制为 NODE_IMPORTANT (MATLAB: dest = temp_node_important(...))
        # 注意：generate_requests_for_source 内部会自动执行 (all_nodes - source) 的排己操作
        requests = generate_requests_for_source(
            source_node=source_node,
            arrive_times=arrive_times,
            all_nodes=NODE_IMPORTANT  # <--- 这里传重要节点，而不是全网节点
        )

        all_requests.extend(requests)

    # 按到达时间排序 (MATLAB: [sorted_arrive_time,sortedIndex] = sort...)
    all_requests.sort(key=lambda r: r['arrival_time'])

    # 重新编号
    for i, req in enumerate(all_requests, 1):
        req['id'] = i

    print(f"\n" + "=" * 70)
    print(f"✓ 总共生成 {len(all_requests)} 个请求")
    print(f"✓ 覆盖源节点: {len(NODE_IMPORTANT)} 个")
    print("=" * 70)

    return all_requests

def save_requests(requests, filename):
    """保存请求到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(requests, f)
    print(f"\n✓ 保存到: {filename}")


if __name__ == '__main__':
    import sys
    import os

    # 确保输出目录存在
    output_dir = './data_output'
    os.makedirs(output_dir, exist_ok=True)

    # 生成 Phase 1 数据
    print("\n生成 Phase 1 数据...")
    phase1_requests = generate_all_requests(T=1000, lamda=0.6, seed=42)
    save_requests(phase1_requests, f'{output_dir}/phase1_requests.pkl')

    # 生成 Phase 3 数据
    print("\n生成 Phase 3 数据...")
    phase3_requests = generate_all_requests(T=1000, lamda=0.6, seed=123)
    save_requests(phase3_requests, f'{output_dir}/phase3_requests.pkl')

    # 验证
    print("\n" + "=" * 70)
    print("验证生成的数据")
    print("=" * 70)

    print(f"\nPhase 1 数据验证:")
    sources = [r['source'] for r in phase1_requests]
    unique_sources = set(sources)

    print(f"  请求数: {len(phase1_requests)}")
    print(f"  唯一源节点: {sorted(unique_sources)}")
    print(f"  源节点数量: {len(unique_sources)}")

    # 检查是否所有源节点都是DC
    non_dc_sources = [s for s in unique_sources if s not in DC_NODES]
    if non_dc_sources:
        print(f"  ❌ 发现非DC源节点: {non_dc_sources}")
    else:
        print(f"  ✅ 所有源节点都是DC节点")

    # 显示示例
    print(f"\n示例请求:")
    for i, req in enumerate(phase1_requests[:3], 1):
        print(f"\n  请求 {i}:")
        print(f"    Source: {req['source']} {'✓DC' if req['source'] in DC_NODES else '✗非DC'}")
        print(f"    Dests: {req['dest']}")
        print(f"    VNFs: {req['vnf']}")
        print(f"    BW: {req['bw_origin']}")

    print("\n✅ 数据生成完成！")