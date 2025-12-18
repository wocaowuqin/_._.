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
DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]

# ✅ 重要节点（汇聚节点，可以作为目标，但不能作为源）
node_important = [16, 21, 22, 24, 25, 26, 27, 28]

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
    生成所有请求

    ✅ 关键修改：源节点从 DC_NODES 中选择，而不是 node_important

    Args:
        T: 时间范围
        lamda: 泊松到达率
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("=" * 70)
    print("生成请求数据")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  时间范围: {T}")
    print(f"  到达率: {lamda}")
    print(f"  DC节点数: {len(DC_NODES)}")
    print(f"  源节点池: DC节点 {DC_NODES[:5]}...")
    print(f"  目标节点池: 所有节点 [1-28]")
    print(f"  VNF链长度: {VNF_CHAIN_LENGTH}")
    print(f"  目标数量: {NUM_DESTINATIONS}")

    all_requests = []

    # ✅ 关键修改：为每个 DC 节点生成请求
    for source_node in DC_NODES:
        print(f"\n为DC节点 {source_node} 生成请求...")

        # 生成到达时间
        arrive_times = generate_poisson_arrivals(T, lamda)
        print(f"  生成了 {len(arrive_times)} 个到达事件")

        # 生成请求
        requests = generate_requests_for_source(
            source_node=source_node,
            arrive_times=arrive_times,
            all_nodes=list(range(1, NUM_NODES + 1))
        )

        all_requests.extend(requests)

    # 按到达时间排序
    all_requests.sort(key=lambda r: r['arrival_time'])

    # 重新编号
    for i, req in enumerate(all_requests, 1):
        req['id'] = i

    print(f"\n" + "=" * 70)
    print(f"✓ 总共生成 {len(all_requests)} 个请求")
    print(f"✓ 平均每个DC节点: {len(all_requests) / len(DC_NODES):.1f} 个请求")
    print("=" * 70)

    return all_requests


def save_requests(requests, filename):
    """保存请求到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(requests, f)
    print(f"\n✓ 保存到: {filename}")


# ... (前面的代码保持不变)

def generate_events_file(requests, output_path, max_time_steps):
    """
    根据请求列表，生成时间轴事件文件
    """
    print(f"正在生成事件文件: {output_path} ...")

    # 初始化时间轴：列表索引即为时间步 t
    # 长度设为最大时间步 + 缓冲
    events_timeline = [{'arrive_event': [], 'leave_event': []} for _ in range(max_time_steps + 200)]

    count = 0
    for req in requests:
        # 获取该请求的到达和离开时间步
        t_arrive = req.get('arrive_time_step')
        t_leave = req.get('leave_time_step')
        rid = req.get('id')

        # 填充到达事件
        if t_arrive is not None and t_arrive < len(events_timeline):
            events_timeline[t_arrive]['arrive_event'].append(rid)

        # 填充离开事件
        if t_leave is not None and t_leave < len(events_timeline):
            events_timeline[t_leave]['leave_event'].append(rid)

        count += 1

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(events_timeline, f)
    print(f"✓ 事件文件生成完毕 (包含 {count} 个请求的调度信息)")


if __name__ == '__main__':
    import sys
    import os

    # 确保输出目录存在
    output_dir = './data/input_dir'  # ⚠️ 注意：请确认您的项目读取路径是 ./data_output 还是 ./data/input_dir
    # 如果您的 main.py 读取的是 data/input_dir，请修改这里！
    # 根据之前的日志，似乎是 ./data/input_dir 或 ./data_output
    # 建议您先确认一下，或者两边都存一下

    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成 Phase 1 数据
    print("\n=== 生成 Phase 1 数据 ===")
    phase1_requests = generate_all_requests(T=1000, lamda=0.6, seed=42)
    save_requests(phase1_requests, f'{output_dir}/phase1_requests.pkl')

    # ✅ 新增：生成对应的事件文件
    generate_events_file(phase1_requests, f'{output_dir}/phase1_events.pkl', max_time_steps=1200)

    # 2. 生成 Phase 3 数据
    print("\n=== 生成 Phase 3 数据 ===")
    phase3_requests = generate_all_requests(T=1000, lamda=0.6, seed=123)
    save_requests(phase3_requests, f'{output_dir}/phase3_requests.pkl')

    # ✅ 新增：生成对应的事件文件
    generate_events_file(phase3_requests, f'{output_dir}/phase3_events.pkl', max_time_steps=1200)

    # 验证... (保持不变)
    print("\n✅ 所有数据(Requests + Events)生成完成！请重新运行 main.py")