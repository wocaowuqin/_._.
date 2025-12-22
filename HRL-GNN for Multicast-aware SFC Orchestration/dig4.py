#!/usr/bin/env python3
"""Phase 1 → Phase 2 数据质量检查"""

import pickle
import numpy as np
import torch
from pathlib import Path

print("=" * 70)
print("🔬 Phase 1 → Phase 2 数据质量检查")
print("=" * 70)

data_path = Path("outputs/expert/expert_data_final.pkl")

if not data_path.exists():
    print(f"❌ 数据文件不存在: {data_path}")
    exit(1)

# 加载数据
with open(data_path, 'rb') as f:
    data = pickle.load(f)

transitions = data.get('success', [])

print(f"\n📊 基本信息:")
print(f"  原始 Transitions: {len(transitions)}")

# ============================================
# 检查 1: 数据格式
# ============================================
print(f"\n✓ 检查 1: 数据格式")

if len(transitions) == 0:
    print("  ❌ 没有数据！")
    exit(1)

sample = transitions[0]
print(f"  第一个 transition 的键: {sample.keys()}")

required_keys = ['state', 'action', 'reward', 'next_state', 'done']
for key in required_keys:
    if key in sample:
        print(f"    ✅ {key}: 存在")
    else:
        print(f"    ❌ {key}: 缺失！")

# ============================================
# 检查 2: State 质量
# ============================================
print(f"\n✓ 检查 2: State 质量")

state_sample = sample.get('state')

if state_sample is None:
    print("  ❌ State 是 None！")
else:
    print(f"  State 类型: {type(state_sample)}")

    if hasattr(state_sample, 'x'):
        # PyG Data 对象
        print(f"  ✅ State 是 PyG Data 对象")
        print(f"    节点特征 x: {state_sample.x.shape}")
        print(f"    边索引 edge_index: {state_sample.edge_index.shape}")

        # 检查节点特征是否全零
        if hasattr(state_sample, 'x'):
            x_mean = state_sample.x.mean().item()
            x_std = state_sample.x.std().item()
            x_max = state_sample.x.max().item()

            print(f"\n    节点特征统计:")
            print(f"      均值: {x_mean:.4f}")
            print(f"      标准差: {x_std:.4f}")
            print(f"      最大值: {x_max:.4f}")

            if x_std < 0.01:
                print(f"      ❌ 节点特征几乎全零或常数！")
                print(f"         这会导致 GNN 无法学习")
            elif x_std < 0.1:
                print(f"      ⚠️  节点特征变化很小")
            else:
                print(f"      ✅ 节点特征正常")

        # 检查是否有 req_vec
        if hasattr(state_sample, 'req_vec'):
            print(f"    ✅ req_vec: {state_sample.req_vec.shape}")

            req_mean = state_sample.req_vec.mean().item()
            req_std = state_sample.req_vec.std().item()

            print(f"      均值: {req_mean:.4f}")
            print(f"      标准差: {req_std:.4f}")

            if req_std < 0.01:
                print(f"      ⚠️  req_vec 几乎全零")
        else:
            print(f"    ⚠️  没有 req_vec")

    else:
        print(f"  ⚠️  State 不是 PyG Data 对象")
        print(f"     这可能导致数据加载问题")

# ============================================
# 检查 3: Action 分布和映射
# ============================================
print(f"\n✓ 检查 3: Action 质量")

# 收集所有 action
all_actions = []
action_formats = set()

for trans in transitions[:100]:  # 检查前 100 个
    action = trans.get('action')

    action_formats.add(str(type(action)))

    if isinstance(action, dict):
        path = action.get('path', [])
        if len(path) > 1:
            for node in path[1:]:
                if isinstance(node, np.integer):
                    node = int(node)

                # 转换为 0-based
                if node >= 1 and node <= 28:
                    node = node - 1

                all_actions.append(node)

    elif isinstance(action, (int, np.integer)):
        all_actions.append(int(action))

print(f"  Action 格式类型: {action_formats}")
print(f"  提取的 Action 数: {len(all_actions)}")

if all_actions:
    print(f"  Action 范围: [{min(all_actions)}, {max(all_actions)}]")

    invalid = [a for a in all_actions if a < 0 or a > 27]
    if invalid:
        print(f"  ❌ 发现 {len(invalid)} 个无效 Action！")
        print(f"     示例: {invalid[:10]}")
    else:
        print(f"  ✅ 所有 Action 都有效")

# ============================================
# 检查 4: State-Action 配对
# ============================================
print(f"\n✓ 检查 4: State-Action 配对")

valid_pairs = 0
invalid_pairs = 0

for i, trans in enumerate(transitions[:100]):
    state = trans.get('state')
    action = trans.get('action')

    if state is None:
        invalid_pairs += 1
        continue

    if not hasattr(state, 'x'):
        invalid_pairs += 1
        continue

    # 提取 action
    if isinstance(action, dict):
        path = action.get('path', [])
        if len(path) <= 1:
            invalid_pairs += 1
            continue

    valid_pairs += 1

print(f"  有效配对: {valid_pairs}/100")
print(f"  无效配对: {invalid_pairs}/100")

if invalid_pairs > 10:
    print(f"  ❌ 过多无效配对（> 10%）")
else:
    print(f"  ✅ 配对质量良好")

# ============================================
# 检查 5: 同一状态的不同 action
# ============================================
print(f"\n✓ 检查 5: 状态多样性")

# 检查是否所有状态都一样
state_hashes = set()

for trans in transitions[:100]:
    state = trans.get('state')

    if state is not None and hasattr(state, 'x'):
        # 简单的哈希（用节点特征的和）
        state_hash = state.x.sum().item()
        state_hashes.add(state_hash)

print(f"  不同状态数: {len(state_hashes)}/100")

if len(state_hashes) < 5:
    print(f"  ❌ 状态几乎完全相同！")
    print(f"     这会导致模型无法区分不同情况")
elif len(state_hashes) < 20:
    print(f"  ⚠️  状态多样性较低")
else:
    print(f"  ✅ 状态多样性良好")

# ============================================
# 总结
# ============================================
print(f"\n" + "=" * 70)
print(f"🎯 诊断总结")
print(f"=" * 70)

issues = []

# 检查节点特征
if hasattr(transitions[0].get('state', {}), 'x'):
    x = transitions[0]['state'].x
    if x.std().item() < 0.01:
        issues.append("节点特征几乎全零")

# 检查状态多样性
if len(state_hashes) < 10:
    issues.append(f"状态多样性过低（只有 {len(state_hashes)} 种）")

# 检查配对
if invalid_pairs > 10:
    issues.append(f"过多无效 State-Action 配对（{invalid_pairs}%）")

if issues:
    print(f"\n❌ 发现以下问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print(f"\n💡 可能的原因:")
    print(f"  1. Phase 1 Expert 收集的数据质量差")
    print(f"  2. State 特征提取有问题")
    print(f"  3. 环境的 reset/step 实现有问题")

    print(f"\n🔧 建议:")
    print(f"  1. 重新运行 Phase 1 收集更多样化的数据")
    print(f"  2. 检查环境的 State 生成逻辑")
    print(f"  3. 确保每个 State 都包含有用的信息")
else:
    print(f"\n✅ 数据质量检查通过")
    print(f"\n如果损失仍然高，可能的原因:")
    print(f"  1. 学习率设置不当")
    print(f"  2. GNN 架构与数据不匹配")
    print(f"  3. 需要更多训练 Epochs")

print(f"=" * 70)