#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成器 - 论文复刻版 (双阶段生成)
---------------------------------------------------------
功能：
1. 生成 Phase 1 (专家训练) 数据集：使用种子 42，时长较长
2. 生成 Phase 3 (RL评估) 数据集：使用种子 123，严格对齐论文 400 间隔
---------------------------------------------------------
"""

import random
import numpy as np
import pickle
import os

# ========== 1. 核心网络配置 (论文/MATLAB一致) ==========
# 8个非DC节点作为流量源/目的
NODE_TRAFFIC_LIST = [16, 21, 22, 24, 25, 26, 27, 28]

# ========== 2. 时间与负载参数 ==========
# 时间间隔 5秒
TIME_INTERVAL = 5.0

# 负载: 0.6 req/间隔 -> 0.12 req/s
#LAMBDA_PER_INTERVAL = 0.6
LAMBDA_PER_INTERVAL = 7.0
LAMBDA_RATE = LAMBDA_PER_INTERVAL / TIME_INTERVAL

# ========== 3. 业务请求参数 ==========
NUM_DESTINATIONS = 5
VNF_CHAIN_LENGTH = 3
VNF_TYPES = 8

MIN_BANDWIDTH = 4
MAX_BANDWIDTH = 8
MEAN_LIFETIME = 3  # 3个时间间隔


def generate_single_request(req_id, source, destinations, vnf_chain, bandwidth,
                            cpu_needs, mem_needs, arrive_time, lifetime):
    """生成单个请求对象"""
    leave_time = arrive_time + lifetime

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
        'arrive_time_step': arrive_time_step,
        'leave_time_step': leave_time_step
    }


def generate_poisson_arrivals(T, lamda):
    """生成泊松到达时间序列"""
    arrivals = []
    time_state = 0
    while time_state < T:
        interval = np.random.exponential(1.0 / lamda)
        time_state += interval
        if time_state < T:
            arrivals.append(time_state)
    return arrivals


def generate_vnf_resources(bandwidth):
    """生成资源需求 (MATLAB系数)"""
    cpu_factor = np.random.rand() * 2.75 + 0.25
    mem_factor = np.random.rand() * 1.75 + 0.25
    cpu = round(bandwidth * cpu_factor)
    mem = round(bandwidth * mem_factor)
    return cpu, mem


def generate_all_requests(num_intervals, lamda, seed=None, phase_name="Unknown"):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 计算总物理时长
    T_duration = num_intervals * TIME_INTERVAL

    print("=" * 60)
    print(f"🚀 生成 {phase_name} 数据")
    print(f"   随机种子: {seed}")
    print(f"   时间间隔: {num_intervals} (总时长 {T_duration}s)")
    print(f"   到达率: {LAMBDA_PER_INTERVAL} req/间隔 ({lamda:.3f} req/s)")
    print("=" * 60)

    all_requests = []

    # 遍历 8 个非 DC 节点作为源
    for source_node in NODE_TRAFFIC_LIST:
        # 1. 泊松流
        arrive_times = generate_poisson_arrivals(T_duration, lamda)

        # 2. 候选目的 (排除自己)
        candidate_dests = [n for n in NODE_TRAFFIC_LIST if n != source_node]

        for arrive_time in arrive_times:
            # 3. 随机目的
            destinations = random.sample(candidate_dests, NUM_DESTINATIONS)
            # 4. 随机VNF
            vnf_chain = random.sample(range(1, VNF_TYPES + 1), VNF_CHAIN_LENGTH)
            # 5. 随机带宽
            bandwidth = random.randint(MIN_BANDWIDTH, MAX_BANDWIDTH)
            # 6. 资源需求
            cpu_needs, mem_needs = [], []
            for _ in vnf_chain:
                c, m = generate_vnf_resources(bandwidth)
                cpu_needs.append(c)
                mem_needs.append(m)

            # 7. 持续时间 (3个间隔 * 5秒 = 15秒均值)
            # MATLAB: lifetime = 1 + exprnd(3-1) -> mean 3
            base_slots = 1 + np.random.exponential(MEAN_LIFETIME - 1)
            while base_slots > 6:  # 截断
                base_slots = 1 + np.random.exponential(MEAN_LIFETIME - 1)

            lifetime_seconds = base_slots * TIME_INTERVAL

            req = generate_single_request(
                req_id=0,
                source=source_node,
                destinations=destinations,
                vnf_chain=vnf_chain,
                bandwidth=bandwidth,
                cpu_needs=cpu_needs,
                mem_needs=mem_needs,
                arrive_time=arrive_time,
                lifetime=lifetime_seconds
            )
            all_requests.append(req)

    # 排序与重编号
    all_requests.sort(key=lambda r: r['arrival_time'])
    for i, req in enumerate(all_requests, 1):
        req['id'] = i

    print(f"✅ {phase_name} 生成完毕: 共 {len(all_requests)} 条请求")
    return all_requests


if __name__ == '__main__':
    output_dir = './data/input_dir'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. 生成 Phase 1 (Expert Training) 数据
    # ---------------------------------------------------------
    # 建议：Phase 1 可以生成更多的数据用于训练，这里设为 800 个间隔 (4000秒)
    # 随机种子：42
    phase1_reqs = generate_all_requests(
        num_intervals=800,  # 比如比测试集多一倍
        lamda=LAMBDA_RATE,
        seed=42,
        phase_name="Phase 1 (Training)"
    )
    with open(f'{output_dir}/phase1_requests.pkl', 'wb') as f:
        pickle.dump(phase1_reqs, f)

    # ---------------------------------------------------------
    # 2. 生成 Phase 3 (RL Evaluation) 数据
    # ---------------------------------------------------------
    # 论文设定：400 个间隔 (2000秒)
    # 随机种子：123
    phase3_reqs = generate_all_requests(
        num_intervals=400,
        lamda=LAMBDA_RATE,
        seed=123,
        phase_name="Phase 3 (Evaluation)"
    )
    with open(f'{output_dir}/phase3_requests.pkl', 'wb') as f:
        pickle.dump(phase3_reqs, f)

    print("\n🎉 所有数据集生成完成！")
    print(f"📂 保存路径: {output_dir}")