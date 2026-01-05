#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试集脚本 (generate_test_data.py)

功能：
1. 生成固定随机种子 (Seed=2024) 的测试请求集，用于最终模型评估。
2. 确保测试集分布与训练集一致 (泊松分布, 节点范围等)。
3. 同时生成对应的事件列表 (Event List)，以便模拟器使用。

使用方法：
    python generate_test_data.py
"""

import os
import pickle
import numpy as np
# 复用 main_generate.py 中的核心生成逻辑
# 请确保 main_generate.py 在同一目录下
from main_generate import generate_all_requests
# 复用 generate_event.py 中的事件处理逻辑
from generate_event import process_single_file

# ========== 配置 ==========
OUTPUT_DIR = './data_output'  # 输出目录
TEST_SEED = 2024  # 测试集专用种子 (必须与训练集不同!)
TEST_DURATION = 200  # 测试集时长 (与训练集保持一致或更短)
LAMBDA_RATE = 0.6  # 到达率 (保持一致)


def main():
    print("=" * 70)
    print("🚀 开始生成测试数据集 (Test Set Generation)")
    print("=" * 70)

    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 生成测试请求 (Requests)
    # 调用 main_generate.py 中的函数
    print(f"\n[1/3] 生成测试请求 (T={TEST_DURATION}, Lambda={LAMBDA_RATE}, Seed={TEST_SEED})...")
    test_requests = generate_all_requests(
        T=TEST_DURATION,
        lamda=LAMBDA_RATE,
        seed=TEST_SEED
    )

    # 3. 保存请求文件
    req_filename = 'test_requests.pkl'
    req_path = os.path.join(OUTPUT_DIR, req_filename)

    with open(req_path, 'wb') as f:
        pickle.dump(test_requests, f)
    print(f"✓ 测试请求已保存至: {req_path}")
    print(f"  - 总请求数: {len(test_requests)}")

    # 4. 生成对应的事件列表 (Events)
    # 模拟器通常需要 event_list 来驱动时间步
    print(f"\n[2/3] 生成测试事件列表...")
    event_filename = 'test_events.pkl'
    process_single_file(req_filename, event_filename)

    # 5. 验证与统计
    print(f"\n[3/3] 验证数据统计...")

    # 简单统计一下源节点分布，确保没有跑偏
    sources = [r['source'] for r in test_requests]
    unique_sources = set(sources)
    print(f"  - 源节点覆盖数: {len(unique_sources)} (预期应覆盖所有重要节点)")

    # 统计 VNF 链长度分布
    chain_lens = [len(r['vnf']) for r in test_requests]
    avg_len = np.mean(chain_lens)
    print(f"  - 平均 VNF 链长度: {avg_len:.2f}")

    print("\n" + "=" * 70)
    print("✅ 测试集生成完毕！")
    print(f"1. 请求文件: {req_path}")
    print(f"2. 事件文件: {os.path.join(OUTPUT_DIR, event_filename)}")
    print("=" * 70)
    print("提示：在 run_inference.py 或 run_eval.py 中，请加载这两个文件进行测试。")


if __name__ == "__main__":
    main()