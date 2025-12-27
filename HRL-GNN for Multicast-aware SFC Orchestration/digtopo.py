"""
测试修复后的 load_topology
"""

import sys
import os

# 确保使用当前目录的代码
sys.path.insert(0, os.getcwd())

# 清除所有 pyc 缓存
import shutil
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))
        print(f"删除缓存: {os.path.join(root, '__pycache__')}")

from utils.config_utils import load_config
import numpy as np

config = load_config('phase3')

print("=" * 80)
print("测试拓扑加载")
print("=" * 80)

# 手动导入并调用 load_topology
from main import load_topology

success = load_topology(config)

if success:
    topo = config['topology']['matrix']
    print(f"\n✅ 加载成功")
    print(f"Topo 形状: {topo.shape}")
    print(f"非零元素: {int(np.sum(topo))}")
    print(f"物理链路数: {int(np.sum(topo)) // 2}")
    print(f"平均度数: {np.sum(topo) / topo.shape[0]:.2f}")

    # 检查是否完全图
    N = topo.shape[0]
    if int(np.sum(topo)) == N * (N - 1):
        print("\n❌❌❌ 还是完全图！")
    else:
        print("\n✅✅✅ 拓扑正常！")
else:
    print("\n❌ 加载失败")

print("=" * 80)