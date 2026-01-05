#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件生成器 - 将请求序列转换为按时间步索引的事件列表
对应 MATLAB: generate_event.m
"""

import pickle
import os
import sys

# 确保路径与 main_generate.py 一致
DATA_DIR = './data/input_dir'


def process_single_file(input_filename, output_filename):
    """
    读取请求文件，生成事件列表 (Event List)
    结构: List[Dict] -> [{'time_step': 0, 'arrive_event': [id1], 'leave_event': []}, ...]
    """
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️  跳过: 未找到 {input_path}")
        return

    print(f"🔄 正在处理: {input_filename} -> {output_filename} ...")

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    if not requests_list:
        print("❌ 请求列表为空！")
        return

    # 1. 找出最大时间步 (确定仿真总长度)
    # MATLAB 逻辑是只记录到最后一次到达的时间步，但为了仿真安全，
    # 我们这里覆盖到最后一次离开的时间步，防止 EventManager 索引越界。
    max_leave_step = 0
    for req in requests_list:
        # 确保读取的是整数时间步
        l_step = int(req['leave_time_step'])
        if l_step > max_leave_step:
            max_leave_step = l_step

    print(f"   ⏱️  最大时间步 (Max Step): {max_leave_step}")

    # 2. 初始化事件列表 (预分配空间)
    # 长度 + 2 是为了防止边界溢出
    event_list = []
    for t in range(max_leave_step + 2):
        event_list.append({
            'time_step': t,
            'arrive_event': [],  # 存储该时刻到达的请求 ID
            'leave_event': []  # 存储该时刻离开的请求 ID
        })

    # 3. 填充事件
    count_arrive = 0
    count_leave = 0

    for req in requests_list:
        req_id = req['id']
        t_arr = int(req['arrive_time_step'])
        t_leave = int(req['leave_time_step'])

        # 记录到达
        if 0 <= t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req_id)
            count_arrive += 1

        # 记录离开
        if 0 <= t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req_id)
            count_leave += 1

    # 4. 保存结果
    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f"✅ 已生成: {output_path}")
    print(f"   统计: 到达事件 {count_arrive} 个, 离开事件 {count_leave} 个")
    print("-" * 50)


def generate_events():
    print("=" * 60)
    print("🚀 生成事件列表 (Event Generation)")
    print(f"📂 工作目录: {DATA_DIR}")
    print("=" * 60)

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 处理 Phase 1 (专家训练数据)
    # 🔥 修改点：去掉文件名前面的 'data/input_dir/'，只保留文件名
    process_single_file('phase1_requests.pkl', 'phase1_events.pkl')

    # 2. 处理 Phase 3 (RL 评估数据)
    # 🔥 修改点：同上
    process_single_file('phase3_requests.pkl', 'phase3_events.pkl')

    print("\n🎉 所有事件列表生成完毕！")

if __name__ == '__main__':
    generate_events()