#!/usr/bin/env python3
"""Phase 2 深度诊断"""

import torch
import pickle
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔬 Phase 2 深度诊断")
print("=" * 70)

# ============================================
# 1. 检查模型配置
# ============================================
print("\n📊 1. 检查模型配置")

model_path = Path("outputs/checkpoints/il_model_final.pth")
if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'policy_net' in checkpoint:
        state_dict = checkpoint['policy_net']

        # 找到输出层
        output_layers = {}
        for key, value in state_dict.items():
            if 'weight' in key and ('output' in key or 'head' in key or 'fc' in key):
                if len(value.shape) >= 2:
                    output_layers[key] = value.shape

        print("  模型输出层:")
        for name, shape in output_layers.items():
            print(f"    {name}: {shape}")

            # 检查输出维度
            output_dim = shape[0]
            if output_dim == 28:
                print(f"      ✅ 输出维度正确: {output_dim}")
            else:
                print(f"      ❌ 输出维度错误: {output_dim} (应该是 28)")

    if 'val_loss' in checkpoint:
        print(f"\n  最终验证损失: {checkpoint['val_loss']:.4f}")

        if checkpoint['val_loss'] > 3.0:
            print(f"    ❌ 损失异常高（> 3.0）")
        elif checkpoint['val_loss'] > 1.0:
            print(f"    ⚠️  损失偏高（> 1.0）")
        else:
            print(f"    ✅ 损失正常（< 1.0）")
else:
    print("  ⚠️  模型文件不存在")

# ============================================
# 2. 分析数据分布
# ============================================
print("\n📊 2. 分析数据分布")

data_path = Path("outputs/expert/expert_data_final.pkl")
if data_path.exists():
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    transitions = data.get('success', [])

    # 提取所有 action（转换后）
    actions = []
    for trans in transitions:
        action = trans.get('action')

        if isinstance(action, dict):
            path = action.get('path', [])
            for node in path[1:]:
                if isinstance(node, np.integer):
                    node = int(node)

                # 转换为 0-based
                if node >= 1 and node <= 28:
                    node = node - 1

                actions.append(node)

    print(f"  总 Action 数: {len(actions)}")
    print(f"  Action 范围: [{min(actions)}, {max(actions)}]")
    print(f"  唯一 Action 数: {len(set(actions))}")

    # Action 分布
    from collections import Counter

    action_counts = Counter(actions)

    print(f"\n  Action 分布（Top 10）:")
    for action, count in action_counts.most_common(10):
        percentage = count / len(actions) * 100
        bar = '█' * int(percentage)
        print(f"    Action {action:2d}: {count:4d} 次 ({percentage:5.2f}%) {bar}")

    # 检查分布是否均匀
    max_count = max(action_counts.values())
    min_count = min(action_counts.values())
    ratio = max_count / max(min_count, 1)

    print(f"\n  分布均匀性:")
    print(f"    最多: {max_count} 次")
    print(f"    最少: {min_count} 次")
    print(f"    比例: {ratio:.2f}:1")

    if ratio > 10:
        print(f"    ⚠️  数据分布严重不均（比例 > 10:1）")
        print(f"       这可能导致模型偏向高频 Action")
    elif ratio > 5:
        print(f"    ⚠️  数据分布不均（比例 > 5:1）")
    else:
        print(f"    ✅ 数据分布相对均匀")

# ============================================
# 3. 计算理论最低损失
# ============================================
print("\n📊 3. 计算理论最低损失")

if actions:
    # 如果模型总是预测最频繁的 action
    most_common_action, most_common_count = action_counts.most_common(1)[0]
    majority_accuracy = most_common_count / len(actions)

    # 交叉熵损失的理论下界
    # 如果模型完美预测，loss = -log(1) = 0
    # 如果模型随机预测（28类），loss = -log(1/28) ≈ 3.33
    # 如果模型总是预测 majority class，loss = -log(majority_accuracy)

    random_baseline_loss = -np.log(1 / 28)
    majority_baseline_loss = -np.log(majority_accuracy)

    print(f"  随机预测 Baseline: {random_baseline_loss:.4f}")
    print(f"  多数类 Baseline: {majority_baseline_loss:.4f}")
    print(f"  完美预测: 0.0000")

    current_loss = checkpoint.get('val_loss', 3.24) if model_path.exists() else 3.24
    print(f"\n  当前验证损失: {current_loss:.4f}")

    if abs(current_loss - random_baseline_loss) < 0.1:
        print(f"    ❌ 损失接近随机猜测（{random_baseline_loss:.2f}）")
        print(f"       模型基本没有学到任何东西！")
    elif current_loss > majority_baseline_loss:
        print(f"    ❌ 损失高于多数类 Baseline（{majority_baseline_loss:.2f}）")
        print(f"       模型表现不如总是预测最频繁的 Action")
    elif current_loss > 1.0:
        print(f"    ⚠️  损失偏高（> 1.0），但优于 Baseline")
    else:
        print(f"    ✅ 损失正常（< 1.0）")

# ============================================
# 4. 检查可能的问题
# ============================================
print("\n" + "=" * 70)
print("🎯 诊断结论")
print("=" * 70)

issues = []

# 检查输出维度
if model_path.exists() and output_layers:
    for name, shape in output_layers.items():
        if shape[0] != 28:
            issues.append(f"模型输出维度 {shape[0]} != 28")

# 检查损失
if current_loss > 3.0:
    if abs(current_loss - random_baseline_loss) < 0.1:
        issues.append("损失等于随机猜测，模型完全没学习")
    else:
        issues.append(f"损失异常高（{current_loss:.2f}）")

# 检查数据分布
if ratio > 10:
    issues.append(f"数据分布严重不均（{ratio:.1f}:1）")

if issues:
    print("\n❌ 发现以下问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n💡 可能的原因:")
    print("  1. GNN 没有正确输出特征")
    print("  2. 学习率设置不当")
    print("  3. 数据预处理有误")
    print("  4. 模型架构与任务不匹配")

    print("\n🔧 建议:")
    if abs(current_loss - random_baseline_loss) < 0.1:
        print("  【紧急】模型完全没学习！")
        print("  1. 检查 GNN 是否正确前向传播")
        print("  2. 检查梯度是否正常更新")
        print("  3. 尝试降低学习率")
        print("  4. 检查数据加载是否正确")
else:
    print("\n✅ 未发现明显问题，但损失仍然偏高")
    print("\n可能需要:")
    print("  1. 增加训练 Epochs（当前 15）")
    print("  2. 调整学习率")
    print("  3. 检查 GNN 架构")

print("=" * 70)