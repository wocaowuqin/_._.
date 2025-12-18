import pickle
import os
import torch
import numpy as np
import sys

# 尝试导入 PyG
try:
    from torch_geometric.data import Data, Batch

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ 未检测到 torch_geometric，只能进行基础检查")


def strict_check(file_path):
    print("=" * 60)
    print(f"🏥 正在进行深度体检: {file_path}")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    success_list = data.get('success', [])
    print(f"✅ 样本数量: {len(success_list)}")

    if len(success_list) == 0:
        print("❌ 样本为空，无法检测！")
        return

    # -----------------------------------------------------------
    # 抽样检查 (检查第 1 个样本)
    # -----------------------------------------------------------
    sample = success_list[0]
    state = sample.get('state')

    print("\n[核心检查 1: State 类型]")
    if HAS_PYG and isinstance(state, Data):
        print(f"  ✅ 类型正确: {type(state)}")
    else:
        print(f"  ❌ 类型错误: {type(state)} (期望是 torch_geometric.data.Data)")
        print("     -> Phase 2 训练器 100% 会报错。")
        return

    print("\n[核心检查 2: 维度核对]")
    # 1. 检查 req_vec
    if hasattr(state, 'req_vec') and state.req_vec is not None:
        shape = state.req_vec.shape
        print(f"  👉 req_vec 形状: {shape}")

        # 严格判定
        if len(shape) == 1 and shape[0] == 24:
            print("     ✅ 完美匹配 (24维)")
        elif len(shape) == 2 and shape[1] == 24:
            print("     ✅ 完美匹配 (Batch=1, 24维)")
        else:
            print(f"     ❌ 维度异常! 期望 [24], 实际 {shape}")
            print("     -> 这就是导致 'mat1 and mat2' 报错的元凶！")
    else:
        print("  ❌ 缺失 req_vec 字段！(模型 forward 必须要有它)")

    # 2. 检查 x (节点特征)
    if hasattr(state, 'x') and state.x is not None:
        print(f"  👉 x (节点特征) : {state.x.shape}")
        if state.x.shape[1] == 17:  # 假设 resource.py 里定义的是 17
            print("     ✅ 节点特征维度 17 (符合预期)")
        else:
            print(f"     ⚠️ 节点特征维度 {state.x.shape[1]} (请确认是否符合 resource.py 定义)")

    # 3. 检查 edge_index
    if hasattr(state, 'edge_index') and state.edge_index is not None:
        print(f"  👉 edge_index : {state.edge_index.shape}")
    else:
        print("  ❌ 缺失 edge_index！(图无法构建)")

    print("-" * 60)
    print("诊断结果:")
    if HAS_PYG and isinstance(state, Data) and hasattr(state, 'req_vec') and (state.req_vec.shape[-1] == 24):
        print("🟢 数据非常健康！Phase 2 应该可以通过。")
        print("   如果 Phase 2 依然报错，请检查 resource.py 里的维度定义是否被改动过。")
    else:
        print("🔴 数据依然有问题，请不要运行 Phase 2，先检查 Phase 1 采集代码。")


if __name__ == "__main__":
    # 自动尝试路径
    paths = [
        "outputs/expert/expert_data_final.pkl",
        "E:/pycharmworkspace/SFC-master/HRL-GNN for Multicast-aware SFC Orchestration/outputs/expert/expert_data_final.pkl"
    ]

    found = False
    for p in paths:
        if os.path.exists(p):
            strict_check(p)
            found = True
            break

    if not found:
        print("❌ 未找到数据文件，请手动修改代码里的路径。")