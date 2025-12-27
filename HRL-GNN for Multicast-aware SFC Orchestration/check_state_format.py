"""
===============================================================================
State 格式检测脚本
===============================================================================

这个脚本会检测你的环境返回的 state 到底是什么格式

===============================================================================
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env


def check_state_format():
    """检测 state 格式"""

    print("=" * 80)
    print("State 格式检测")
    print("=" * 80)

    # 加载配置
    try:
        config = load_config('phase3')
        print("✅ 配置加载成功\n")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return

    # 创建环境
    try:
        env = SFC_HIRL_Env(config, use_gnn=True)
        print("✅ 环境创建成功\n")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # Reset 并获取 state
    print("-" * 80)
    print("1. 调用 env.reset()")
    print("-" * 80)

    reset_result = env.reset()

    print(f"返回值类型: {type(reset_result)}")
    print(f"返回值: {reset_result}\n")

    # 提取 state
    if isinstance(reset_result, tuple):
        state = reset_result[0]
        print(f"✅ reset() 返回 tuple，提取第一个元素")
        print(f"   State 类型: {type(state)}")
    else:
        state = reset_result
        print(f"✅ reset() 直接返回 state")
        print(f"   State 类型: {type(state)}")

    print("\n" + "-" * 80)
    print("2. 检测 State 结构")
    print("-" * 80)

    # 检测是否是 Tensor
    if isinstance(state, torch.Tensor):
        print("✅ State 是 torch.Tensor")
        print(f"   Shape: {state.shape}")
        print(f"   Dtype: {state.dtype}")
        print(f"   Device: {state.device}")
        print(f"   维度: {state.dim()}D")
        print(f"   前5个值: {state.flatten()[:5]}")

    # 检测是否是 PyG Data
    elif hasattr(state, 'x'):
        print("✅ State 是 PyG Data 对象")
        print(f"   Has x: {hasattr(state, 'x')}")
        print(f"   Has edge_index: {hasattr(state, 'edge_index')}")
        print(f"   Has edge_attr: {hasattr(state, 'edge_attr')}")
        print(f"   Has req_vec: {hasattr(state, 'req_vec')}")
        print(f"   Has batch: {hasattr(state, 'batch')}")

        if hasattr(state, 'x'):
            print(f"\n   x.shape: {state.x.shape}")
            print(f"   x.dtype: {state.x.dtype}")

        if hasattr(state, 'edge_index'):
            print(f"\n   edge_index.shape: {state.edge_index.shape}")
            print(f"   edge_index 前5条边: {state.edge_index[:, :5]}")

        if hasattr(state, 'edge_attr'):
            print(f"\n   edge_attr.shape: {state.edge_attr.shape}")

        if hasattr(state, 'req_vec'):
            print(f"\n   req_vec.shape: {state.req_vec.shape}")

    # 检测是否是字典
    elif isinstance(state, dict):
        print("✅ State 是字典")
        print(f"   Keys: {list(state.keys())}")
        for key, val in state.items():
            print(f"   {key}: {type(val)} - {val.shape if hasattr(val, 'shape') else val}")

    # 未知格式
    else:
        print("❌ State 是未知格式")
        print(f"   Type: {type(state)}")
        print(f"   Dir: {dir(state)[:10]}...")  # 只显示前10个属性

    print("\n" + "-" * 80)
    print("3. 测试 Step")
    print("-" * 80)

    try:
        # 随机动作
        high_act = 0
        low_act = 0

        env.step_high_level(high_act)
        step_result = env.step_low_level(low_act)

        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
        else:
            next_state, reward, done, info = step_result
            next_state = next_state

        print(f"✅ Step 成功")
        print(f"   Next State 类型: {type(next_state)}")

        # 检查 next_state 是否和 state 一致
        if type(next_state) == type(state):
            print(f"   ✅ Next State 和 State 类型一致")
        else:
            print(f"   ⚠️  Next State 类型不一致: {type(next_state)} vs {type(state)}")

    except Exception as e:
        print(f"❌ Step 失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("检测完成！")
    print("=" * 80)

    # 生成建议
    print("\n📝 建议：\n")

    if isinstance(state, torch.Tensor):
        print("你的 state 是 Tensor，需要：")
        print("1. 修改 agent.py 中的 state 处理逻辑")
        print("2. 不要尝试访问 state.x, state.edge_index 等属性")
        print("3. 直接使用 state 作为特征输入")
        print("\n修改示例：")
        print("""
# 在 _select_action_phase3_goal_conditioned 中
if isinstance(state, torch.Tensor):
    # 直接使用 state
    state_feat = state.unsqueeze(0) if state.dim() == 1 else state
    # 不使用图结构
    edge_index = None
    edge_attr = None
        """)

    elif hasattr(state, 'x'):
        print("你的 state 是 PyG Data，当前代码应该可以正常工作")
        if not hasattr(state, 'edge_index'):
            print("⚠️  但是缺少 edge_index，需要检查环境的图构建逻辑")

    else:
        print("你的 state 格式未知，需要手动检查环境代码")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_state_format()