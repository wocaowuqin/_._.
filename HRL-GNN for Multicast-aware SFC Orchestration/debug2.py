#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1/2/3 数据流诊断工具
检查：
1. Phase 1 专家数据质量
2. Phase 2 数据结构
3. Phase 2 -> Phase 3 数据一致性
4. Encoder 冻结策略是否合理
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ============================================================================
# 1. Phase 1 数据质量分析
# ============================================================================
def analyze_phase1_quality(data_path="outputs/expert/expert_data_final.pkl"):
    print_section("Phase 1 专家数据质量分析")

    if not Path(data_path).exists():
        print(f"❌ 文件不存在: {data_path}")
        return None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    success_samples = data.get('success', [])
    fail_samples = data.get('fail', [])

    total = len(success_samples) + len(fail_samples)
    success_rate = len(success_samples) / total * 100 if total > 0 else 0

    print(f"📊 总体统计:")
    print(f"  总样本数: {total}")
    print(f"  成功样本: {len(success_samples)} ({success_rate:.1f}%)")
    print(f"  失败样本: {len(fail_samples)} ({100 - success_rate:.1f}%)")

    if success_rate < 50:
        print(f"\n⚠️  警告：专家成功率过低 ({success_rate:.1f}%)")
        print("   可能原因：")
        print("   1. 网络资源不足（容量设置过小）")
        print("   2. 请求需求过大（bw/cpu/mem 过高）")
        print("   3. 专家算法本身问题")

    # 分析成功样本的数据质量
    if success_samples:
        print(f"\n📈 成功样本分析 (前100个):")

        # 检查关键字段
        missing_fields = defaultdict(int)
        state_types = defaultdict(int)
        action_types = defaultdict(int)

        for i, sample in enumerate(success_samples[:100]):
            # 检查必需字段
            required_fields = ['state', 'request', 'high_action', 'action']
            for field in required_fields:
                if field not in sample:
                    missing_fields[field] += 1

            # 检查 state 类型
            if 'state' in sample:
                state = sample['state']
                state_type = type(state).__name__
                state_types[state_type] += 1

                # 详细检查 PyG Data
                if hasattr(state, 'x'):
                    if i == 0:  # 只打印第一个
                        print(f"\n  样本 0 的 State 结构:")
                        print(f"    类型: {state_type}")
                        print(f"    x.shape: {state.x.shape}")
                        print(f"    edge_index.shape: {state.edge_index.shape}")
                        if hasattr(state, 'edge_attr'):
                            print(f"    edge_attr.shape: {state.edge_attr.shape}")
                        if hasattr(state, 'req_vec'):
                            print(f"    req_vec.shape: {state.req_vec.shape}")

            # 检查 action 类型
            if 'action' in sample:
                action_type = type(sample['action']).__name__
                action_types[action_type] += 1

        if missing_fields:
            print(f"\n  ⚠️  缺失字段统计:")
            for field, count in missing_fields.items():
                print(f"    {field}: {count}/100 样本缺失")

        print(f"\n  State 类型分布:")
        for stype, count in state_types.items():
            print(f"    {stype}: {count}")

        print(f"\n  Action 类型分布:")
        for atype, count in action_types.items():
            print(f"    {atype}: {count}")

    # 分析失败原因
    if fail_samples:
        print(f"\n📉 失败样本分析:")

        failure_reasons = defaultdict(int)
        for sample in fail_samples[:100]:
            reason = sample.get('reason', 'unknown')
            failure_reasons[reason] += 1

        print(f"  失败原因分布:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    return {
        'total': total,
        'success': len(success_samples),
        'fail': len(fail_samples),
        'success_rate': success_rate,
        'success_samples': success_samples,
        'fail_samples': fail_samples
    }


# ============================================================================
# 2. Phase 2 数据结构检查
# ============================================================================
def analyze_phase2_data_structure(phase1_data):
    print_section("Phase 2 训练数据结构检查")

    if not phase1_data or not phase1_data['success_samples']:
        print("❌ 没有可用的训练数据")
        return None

    samples = phase1_data['success_samples']

    print(f"📦 检查 Phase 2 DataLoader 输入格式:")

    # 模拟 collate_fn
    batch_size = min(4, len(samples))
    batch = samples[:batch_size]

    print(f"\n  测试批次大小: {batch_size}")

    states = []
    actions = []

    for i, item in enumerate(batch):
        state = item.get('state')
        action = item.get('dest_idx', item.get('high_action'))

        if state is None:
            print(f"    ❌ 样本 {i}: state 为 None")
            continue

        if action is None:
            print(f"    ❌ 样本 {i}: action 为 None")
            continue

        states.append(state)
        actions.append(action)

        if i == 0:
            print(f"\n  样本 0 详细信息:")
            print(f"    state 类型: {type(state)}")
            print(f"    action: {action} (type: {type(action)})")

            if hasattr(state, 'x'):
                print(f"    state.x dtype: {state.x.dtype}")
                print(f"    state.x device: {state.x.device}")

    if not states:
        print("  ❌ 没有有效的训练样本！")
        return None

    # 检查 Batch 兼容性
    try:
        from torch_geometric.data import Batch
        graph_batch = Batch.from_data_list(states)
        print(f"\n  ✅ PyG Batch 成功创建:")
        print(f"    batch.x.shape: {graph_batch.x.shape}")
        print(f"    batch.edge_index.shape: {graph_batch.edge_index.shape}")
        print(f"    batch.batch.shape: {graph_batch.batch.shape}")
    except Exception as e:
        print(f"  ❌ Batch 创建失败: {e}")
        return None

    return {
        'valid_samples': len(states),
        'batch_compatible': True
    }


# ============================================================================
# 3. Phase 2 -> Phase 3 数据一致性
# ============================================================================
def check_phase2_to_phase3_consistency():
    print_section("Phase 2 -> Phase 3 数据一致性检查")

    # 检查模型保存
    model_paths = {
        'il_model_final': 'outputs/checkpoints/il_model_final.pth',
        'il_model_encoder': 'outputs/checkpoints/il_model_encoder.pth',
    }

    print(f"📁 检查保存的模型文件:")

    saved_models = {}
    for name, path in model_paths.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
            try:
                state_dict = torch.load(path, map_location='cpu')
                saved_models[name] = state_dict

                # 检查参数数量
                if isinstance(state_dict, dict):
                    if 'policy_net' in state_dict:
                        params = state_dict['policy_net']
                    else:
                        params = state_dict

                    param_count = sum(p.numel() for p in params.values() if isinstance(p, torch.Tensor))
                    print(f"      参数量: {param_count:,}")
            except Exception as e:
                print(f"      ⚠️  加载失败: {e}")
        else:
            print(f"  ❌ {name}: 文件不存在")

    # 检查 Phase 3 能否正确加载
    if 'il_model_final' in saved_models:
        print(f"\n🔄 模拟 Phase 3 加载:")

        try:
            # 检查键结构
            state_dict = saved_models['il_model_final']

            if 'policy_net' in state_dict:
                print(f"  ✅ 找到 'policy_net' 键")
                inner_dict = state_dict['policy_net']
            else:
                print(f"  ⚠️  直接使用 state_dict (无 'policy_net' 包装)")
                inner_dict = state_dict

            # 检查关键层
            gnn_keys = [k for k in inner_dict.keys() if 'gnn' in k]
            encoder_keys = [k for k in inner_dict.keys() if 'encoder' in k]

            print(f"  GNN 相关参数: {len(gnn_keys)}")
            print(f"  Encoder 相关参数: {len(encoder_keys)}")

            if gnn_keys:
                print(f"    示例: {gnn_keys[:3]}")

        except Exception as e:
            print(f"  ❌ 检查失败: {e}")

    return saved_models


# ============================================================================
# 4. Encoder 冻结策略分析
# ============================================================================
def analyze_encoder_freezing_strategy():
    print_section("Encoder 冻结策略分析")

    print("🤔 Encoder 冻结的利弊分析:\n")

    print("✅ 冻结 Encoder 的优点:")
    print("  1. 保留 Phase 2 学到的特征提取能力")
    print("  2. 减少训练参数，加快 Phase 3 训练速度")
    print("  3. 防止 RL 阶段破坏已学习的特征表示")
    print("  4. 更稳定的训练过程")

    print("\n❌ 冻结 Encoder 的缺点:")
    print("  1. 如果 Phase 2 学习不充分，会限制 Phase 3 性能")
    print("  2. 无法适应 Phase 3 新的数据分布")
    print("  3. 可能导致特征表示与 Q 值不匹配")

    print("\n💡 推荐策略:")
    print("  1. 【当前策略】Phase 2 冻结 -> 适合 Phase 2 数据质量高的情况")
    print("  2. 【渐进解冻】Phase 3 初期冻结，后期部分解冻")
    print("  3. 【完全微调】Phase 3 使用极小学习率微调全部参数")
    print("  4. 【两阶段训练】Phase 3 先冻结训练，再解冻微调")

    print("\n🔍 判断是否应该冻结的标准:")
    print("  - Phase 2 训练 loss 是否收敛？")
    print("  - Phase 2 验证准确率是否 > 70%？")
    print("  - Phase 1 专家数据成功率是否 > 60%？")
    print("  如果都满足 -> 推荐冻结")
    print("  否则 -> 推荐解冻或渐进解冻")


# ============================================================================
# 5. Phase 1 专家效果差的原因诊断
# ============================================================================
def diagnose_phase1_poor_performance(phase1_data):
    print_section("Phase 1 专家效果差的原因诊断")

    if not phase1_data:
        print("❌ 没有 Phase 1 数据")
        return

    success_rate = phase1_data['success_rate']

    print(f"📊 当前专家成功率: {success_rate:.1f}%\n")

    if success_rate < 30:
        print("🔴 严重问题 (< 30%):")
        print("  可能原因：")
        print("  1. 资源容量配置过小")
        print("     检查: config['capacities']['cpu/memory/bandwidth']")
        print("  2. 请求需求过大")
        print("     检查: request['cpu_origin'], request['memory_origin'], request['bw_origin']")
        print("  3. 拓扑结构问题")
        print("     检查: DC 节点数量、网络连通性")

    elif success_rate < 60:
        print("🟡 中等问题 (30-60%):")
        print("  可能原因：")
        print("  1. 负载过高（时间步内请求过多）")
        print("  2. 专家算法策略不够优")
        print("  3. Backup Policy 兜底不够强")

    else:
        print("🟢 可接受范围 (> 60%):")
        print("  专家效果尚可，可以继续训练")

    # 具体分析失败样本
    if phase1_data['fail_samples']:
        print(f"\n🔍 失败样本深度分析 (前10个):")

        for i, sample in enumerate(phase1_data['fail_samples'][:10]):
            req = sample.get('request', {})
            reason = sample.get('reason', 'unknown')

            if i == 0:
                print(f"\n  失败样本 0:")
                print(f"    失败原因: {reason}")
                print(f"    请求 ID: {req.get('id', 'N/A')}")
                print(f"    源节点: {req.get('source', 'N/A')}")
                print(f"    目标数量: {len(req.get('dest', []))}")
                print(f"    VNF 数量: {len(req.get('vnf', []))}")
                print(f"    带宽需求: {req.get('bw_origin', 'N/A')}")

                cpu_reqs = req.get('cpu_origin', [])
                mem_reqs = req.get('memory_origin', [])

                if cpu_reqs:
                    print(f"    CPU 需求: avg={np.mean(cpu_reqs):.2f}, max={np.max(cpu_reqs):.2f}")
                if mem_reqs:
                    print(f"    MEM 需求: avg={np.mean(mem_reqs):.2f}, max={np.max(mem_reqs):.2f}")


# ============================================================================
# 6. 生成可视化报告
# ============================================================================
def generate_visualization_report(phase1_data):
    print_section("生成可视化报告")

    if not phase1_data:
        print("❌ 没有数据可视化")
        return

    try:
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 成功 vs 失败
        ax1 = axes[0, 0]
        labels = ['Success', 'Fail']
        sizes = [phase1_data['success'], phase1_data['fail']]
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Phase 1 Expert Success Rate')

        # 2. 失败原因分布
        if phase1_data['fail_samples']:
            ax2 = axes[0, 1]
            failure_reasons = defaultdict(int)
            for sample in phase1_data['fail_samples']:
                reason = sample.get('reason', 'unknown')
                failure_reasons[reason] += 1

            reasons = list(failure_reasons.keys())
            counts = list(failure_reasons.values())

            ax2.barh(reasons, counts, color='#e74c3c')
            ax2.set_xlabel('Count')
            ax2.set_title('Failure Reasons Distribution')

        # 3. 请求大小分布
        ax3 = axes[1, 0]
        dest_counts = []
        for sample in phase1_data['success_samples'][:100]:
            req = sample.get('request', {})
            dest_counts.append(len(req.get('dest', [])))

        ax3.hist(dest_counts, bins=10, color='#3498db', alpha=0.7)
        ax3.set_xlabel('Number of Destinations')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Request Size Distribution (Success)')

        # 4. VNF 链长度分布
        ax4 = axes[1, 1]
        vnf_counts = []
        for sample in phase1_data['success_samples'][:100]:
            req = sample.get('request', {})
            vnf_counts.append(len(req.get('vnf', [])))

        ax4.hist(vnf_counts, bins=10, color='#9b59b6', alpha=0.7)
        ax4.set_xlabel('VNF Chain Length')
        ax4.set_ylabel('Frequency')
        ax4.set_title('VNF Chain Distribution (Success)')

        plt.tight_layout()

        output_path = 'outputs/phase1_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 可视化报告已保存: {output_path}")

    except Exception as e:
        print(f"  ⚠️  可视化生成失败: {e}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("🔍 HRL-GNN Phase 数据流诊断工具\n")

    # 1. Phase 1 质量分析
    phase1_data = analyze_phase1_quality()

    # 2. Phase 2 数据结构
    if phase1_data:
        analyze_phase2_data_structure(phase1_data)

    # 3. Phase 2->3 一致性
    check_phase2_to_phase3_consistency()

    # 4. Encoder 冻结策略
    analyze_encoder_freezing_strategy()

    # 5. Phase 1 效果诊断
    if phase1_data:
        diagnose_phase1_poor_performance(phase1_data)

    # 6. 可视化
    if phase1_data:
        generate_visualization_report(phase1_data)

    print("\n" + "=" * 80)
    print("✅ 诊断完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()