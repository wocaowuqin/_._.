#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据类型检测脚本 - 诊断 Phase 1/2/3 的数据类型问题
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys


def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_type(obj, name, depth=0):
    """递归检查对象类型"""
    indent = "  " * depth

    if isinstance(obj, dict):
        print(f"{indent}📦 {name}: dict (keys={len(obj)})")
        for key, val in list(obj.items())[:5]:  # 只显示前5个
            check_type(val, f"{name}['{key}']", depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}📋 {name}: {type(obj).__name__} (len={len(obj)})")
        if len(obj) > 0:
            check_type(obj[0], f"{name}[0]", depth + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}🔢 {name}: np.ndarray shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}🔥 {name}: torch.Tensor shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{indent}⚙️  {name}: {type(obj).__name__}")


def check_phase1_data(data_path):
    """检查 Phase 1 专家数据"""
    print_separator("Phase 1 Expert Data")

    if not Path(data_path).exists():
        print(f"❌ 文件不存在: {data_path}")
        return

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ 文件加载成功: {data_path}")
        print(f"顶层类型: {type(data)}")

        if isinstance(data, dict):
            print(f"\n字典键: {list(data.keys())}")

            # 检查 success 样本
            if 'success' in data:
                samples = data['success']
                print(f"\n成功样本数: {len(samples)}")

                if len(samples) > 0:
                    print("\n📊 检查第一个成功样本:")
                    sample = samples[0]
                    check_type(sample, "sample", depth=1)

                    # 详细检查关键字段
                    if 'state' in sample:
                        state = sample['state']
                        print("\n  🔍 State 详细信息:")
                        if hasattr(state, 'x'):
                            print(f"    x: {state.x.dtype if hasattr(state.x, 'dtype') else type(state.x)}")
                        if hasattr(state, 'req_vec'):
                            print(
                                f"    req_vec: {state.req_vec.dtype if hasattr(state.req_vec, 'dtype') else type(state.req_vec)}")

        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            if len(data) > 0:
                check_type(data[0], "data[0]", depth=1)

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_backup_policy_output():
    """模拟检查 BackupPolicy 输出"""
    print_separator("BackupPolicy Output Simulation")

    from envs.modules.sfc_backup_system.utils import build_hvt_from_placement

    # 测试 1: placement dict
    placement_dict = {0: 5, 1: 10}
    n, K_vnf = 28, 8

    print("\n测试 1: placement dict -> numpy array")
    print(f"  输入: {placement_dict}")
    hvt = build_hvt_from_placement(placement_dict, n, K_vnf)
    print(f"  输出: shape={hvt.shape}, dtype={hvt.dtype}")
    print(f"  非零元素: {np.count_nonzero(hvt)}")

    # 测试 2: 累加测试
    print("\n测试 2: numpy array 累加")
    tree_hvt = np.zeros((n, K_vnf), dtype=np.float32)
    print(f"  tree_hvt dtype: {tree_hvt.dtype}")
    print(f"  plan_hvt dtype: {hvt.dtype}")

    try:
        result = tree_hvt + hvt
        print(f"  ✅ 累加成功! result dtype: {result.dtype}")
    except Exception as e:
        print(f"  ❌ 累加失败: {e}")

        # 尝试修复
        print("\n  尝试修复:")
        hvt_fixed = hvt.astype(np.float32)
        print(f"    hvt_fixed dtype: {hvt_fixed.dtype}")
        result = tree_hvt + hvt_fixed
        print(f"    ✅ 修复后成功! result dtype: {result.dtype}")


def check_env_state_init():
    """检查环境初始化的数据类型"""
    print_separator("Environment State Initialization")

    n, L, K_vnf = 28, 45, 8

    print("初始化 current_tree:")
    tree = {
        'tree': np.zeros(L, dtype=np.float32),
        'hvt': np.zeros((n, K_vnf), dtype=np.float32),
        'paths_map': {}
    }

    print(f"  tree dtype: {tree['tree'].dtype}")
    print(f"  hvt dtype: {tree['hvt'].dtype}")

    print("\n模拟 plan 数据:")
    # 模拟两种可能的 plan

    # 情况 1: hvt 是 numpy array (正确)
    plan1 = {
        'hvt': np.zeros((n, K_vnf), dtype=np.int32)  # ❌ 可能是 int32
    }

    # 情况 2: hvt 是 dict (错误)
    plan2 = {
        'hvt': {0: 1, 1: 2}  # ❌ dict 格式
    }

    print("\n情况 1: hvt 是 numpy int32")
    print(f"  plan1['hvt'] dtype: {plan1['hvt'].dtype}")
    try:
        result = tree['hvt'] + plan1['hvt']
        print(f"  ✅ 累加成功! dtype: {result.dtype}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    print("\n情况 2: hvt 是 dict")
    print(f"  plan2['hvt'] type: {type(plan2['hvt'])}")
    try:
        result = tree['hvt'] + plan2['hvt']
        print(f"  ✅ 累加成功! dtype: {result.dtype}")
    except Exception as e:
        print(f"  ❌ 失败 (预期): {e}")


def main():
    print("🔍 HRL-GNN 数据类型检测工具")

    # 检查 Phase 1 数据
    expert_data_path = "outputs/expert/expert_data_final.pkl"
    check_phase1_data(expert_data_path)

    # 检查 BackupPolicy 输出
    check_backup_policy_output()

    # 检查环境初始化
    check_env_state_init()

    print("\n" + "=" * 80)
    print("✅ 检测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()