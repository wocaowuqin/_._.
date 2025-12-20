#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试环境 dtype 安全性"""

import numpy as np
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs.sfc_env import SFC_HIRL_Env
from utils.config_utils import load_config

print("\n" + "=" * 70)
print("🧪 环境 dtype 安全性测试")
print("=" * 70 + "\n")

# ✅ 正确的调用方式
print("📋 Step 1: 加载配置...")
try:
    config = load_config(phase='phase3')  # 传入 phase 名称
    print("✅ 配置加载成功\n")
except Exception as e:
    print(f"❌ 配置加载失败: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 创建环境
print("📋 Step 2: 创建环境...")
try:
    env = SFC_HIRL_Env(config, use_gnn=True)
    print("✅ 环境创建成功\n")
except Exception as e:
    print(f"❌ 环境创建失败: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 测试 reset
print("📋 Step 3: 测试 reset()...")
try:
    obs, info = env.reset()
    print(f"✅ reset() 成功")
    print(f"   current_tree['hvt'].dtype: {env.current_tree['hvt'].dtype}")
    print(f"   current_tree['hvt'].shape: {env.current_tree['hvt'].shape}")

    # 验证 dtype
    assert env.current_tree['hvt'].dtype == np.float32, "❌ dtype 不是 float32！"
    print("   ✅ dtype 验证通过\n")
except Exception as e:
    print(f"❌ reset() 失败: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 测试 step_low_level
print("📋 Step 4: 测试 step_low_level()...")
try:
    action = 0  # 测试动作
    obs, reward, done, truncated, info = env.step_low_level(action)
    print(f"✅ step_low_level() 成功")
    print(f"   reward: {reward}")
    print(f"   done: {done}")
    print(f"   truncated: {truncated}\n")
except Exception as e:
    print(f"❌ step_low_level() 失败: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 测试多次 step（确保 dtype 稳定）
print("📋 Step 5: 测试连续 step（dtype 稳定性）...")
try:
    for i in range(5):
        action = i % env.n
        obs, reward, done, truncated, info = env.step_low_level(action)

        # 每次都验证 dtype
        assert env.current_tree['hvt'].dtype == np.float32

    print(f"✅ 连续 5 次 step，dtype 保持稳定\n")
except Exception as e:
    print(f"❌ 连续 step 失败: {e}\n")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("🎉 所有测试通过！")
print("=" * 70)