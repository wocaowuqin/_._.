"""
诊断 Paths 解析过程
"""

import sys
sys.path.append('.')

import scipy.io
import numpy as np
from pathlib import Path

mat_path = Path('data/input_dir/US_Backbone_path.mat')
mat_data = scipy.io.loadmat(str(mat_path))

print("=" * 80)
print("Paths 结构分析")
print("=" * 80)

paths_matrix = mat_data['Paths']
print(f"\nPaths 矩阵形状: {paths_matrix.shape}")
print(f"Paths 数据类型: {paths_matrix.dtype}")

# 采样几个 cell 看结构
sample_cells = [
    (0, 5), (0, 10), (0, 27),
    (5, 10), (10, 15), (15, 27)
]

print("\n样本 Paths 内容:")
for i, j in sample_cells:
    cell = paths_matrix[i, j]
    print(f"\nPaths[{i}, {j}]:")
    print(f"  类型: {type(cell)}")

    if hasattr(cell, 'dtype'):
        print(f"  dtype: {cell.dtype}")
        if cell.dtype.names:
            print(f"  字段名: {cell.dtype.names}")

            if 'paths' in cell.dtype.names:
                paths_array = cell['paths']
                print(f"  paths 类型: {type(paths_array)}")
                print(f"  paths 形状: {paths_array.shape if hasattr(paths_array, 'shape') else 'N/A'}")

                if isinstance(paths_array, np.ndarray):
                    # 打印实际路径
                    if paths_array.ndim == 1:
                        paths_array = paths_array[np.newaxis, :]

                    print(f"  包含 {paths_array.shape[0]} 条路径:")
                    for k, path in enumerate(paths_array[:3]):  # 只打印前3条
                        nodes = path[path > 0] - 1
                        print(f"    路径{k+1}: {nodes} (长度{len(nodes)})")

# 统计总共能提取多少条边
print("\n" + "=" * 80)
print("统计边的提取")
print("=" * 80)

edges = set()
N = paths_matrix.shape[0]

for i in range(N):
    for j in range(N):
        if i == j:
            continue

        cell = paths_matrix[i, j]

        if not hasattr(cell, 'dtype'):
            continue
        if cell.dtype.names is None:
            continue
        if 'paths' not in cell.dtype.names:
            continue

        paths_array = cell['paths']
        if not isinstance(paths_array, np.ndarray):
            continue

        if paths_array.ndim == 1:
            paths_array = paths_array[np.newaxis, :]

        for path in paths_array:
            nodes = path[path > 0] - 1
            if len(nodes) < 2:
                continue

            for k in range(len(nodes) - 1):
                u, v = int(nodes[k]), int(nodes[k + 1])
                if 0 <= u < N and 0 <= v < N:
                    edges.add(tuple(sorted([u, v])))

print(f"\n从 Paths 提取的边数: {len(edges)}")
print(f"期望边数 (US Backbone): ~100-200")

if len(edges) < 50:
    print("\n❌ 提取的边太少！可能的原因:")
    print("  1. Paths 数据不完整")
    print("  2. 索引转换错误 (MATLAB 1-based vs Python 0-based)")
    print("  3. 只使用了部分路径")

# 打印一些提取到的边
print(f"\n提取到的边样本 (前20条):")
for i, (u, v) in enumerate(sorted(edges)[:20]):
    print(f"  {i+1}. ({u}, {v})")

print("\n" + "=" * 80)