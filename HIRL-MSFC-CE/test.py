"""
Phase 3 问题诊断
================

检查 Phase 3 为什么不训练
"""

import logging
from pathlib import Path

print("=" * 70)
print("Phase 3 问题诊断")
print("=" * 70)

# 🔥 修改为你的实际路径
base_dir = Path(r"E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE\generate_requests_depend_on_poisson\data_output")

# 检查 1：Phase 3 数据文件
print("\n1️⃣ 检查 Phase 3 数据文件")
print("-" * 70)

phase3_files = [
    "phase3_requests.pkl",
    "phase3_events.pkl"
]

for filename in phase3_files:
    filepath = base_dir / filename
    if filepath.exists():
        print(f"✅ {filename} 存在")
    else:
        print(f"❌ {filename} 不存在")

# 检查 2：环境是否有 load_dataset 方法
print("\n2️⃣ 检查环境类")
print("-" * 70)

try:
    import hyperparameters as H
    from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN

    # 创建测试环境
    env = SFC_HIRL_Env_GNN(
        input_dir=H.INPUT_DIR,
        topo=H.TOPOLOGY_MATRIX,
        dc_nodes=H.DC_NODES,
        capacities=H.CAPACITIES,
        use_gnn=True
    )

    if hasattr(env, 'load_dataset'):
        print(f"✅ 环境有 load_dataset 方法")
    else:
        print(f"❌ 环境没有 load_dataset 方法")
        print(f"   → Phase 3 会跳过数据加载")

    if hasattr(env, 'request_list'):
        print(f"✅ 环境有 request_list 属性")
    else:
        print(f"❌ 环境没有 request_list 属性")
        print(f"   → 课程学习无法初始化")

except Exception as e:
    print(f"❌ 环境创建失败: {e}")

# 检查 3：Agent 是否有 RL 模式
print("\n3️⃣ 检查 Agent 类")
print("-" * 70)

try:
    from hirl_sfc_agent_gnn import Agent_SFC_GNN

    # 检查方法
    methods_to_check = [
        'switch_to_rl_mode',
        'select_action',
        'store',
        'update',
        'get_epsilon'
    ]

    for method in methods_to_check:
        if hasattr(Agent_SFC_GNN, method):
            print(f"✅ Agent 有 {method} 方法")
        else:
            print(f"❌ Agent 缺少 {method} 方法")

except Exception as e:
    print(f"❌ Agent 导入失败: {e}")

# 检查 4：输出目录
print("\n4️⃣ 检查输出目录")
print("-" * 70)

output_dir = base_dir / "out_hirl" / "three_phase_training"

if output_dir.exists():
    print(f"✅ 输出目录存在: {output_dir}")

    # 检查子目录
    for phase in ['phase1', 'phase2', 'phase3']:
        phase_dir = output_dir / phase
        if phase_dir.exists():
            files = list(phase_dir.glob("*"))
            print(f"  ✅ {phase}/ 存在 ({len(files)} 个文件)")
        else:
            print(f"  ❌ {phase}/ 不存在")
else:
    print(f"❌ 输出目录不存在: {output_dir}")

# 检查 5：日志文件
print("\n5️⃣ 检查日志")
print("-" * 70)

log_file = base_dir / "out_hirl" / "three_phase_training.log"

if log_file.exists():
    print(f"✅ 日志文件存在: {log_file}")

    # 搜索 Phase 3 关键词
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        keywords = [
            "Phase 3",
            "强化学习微调",
            "Episode 1/",
            "RL 微调"
        ]

        found_any = False
        for keyword in keywords:
            if keyword in content:
                print(f"  ✅ 找到关键词: '{keyword}'")
                found_any = True

        if not found_any:
            print(f"  ❌ 没有找到 Phase 3 相关日志")
            print(f"  → Phase 3 可能没有开始运行")

    except Exception as e:
        print(f"  ❌ 读取日志失败: {e}")
else:
    print(f"❌ 日志文件不存在")

# 总结
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)

print("""
可能的问题：

1. Phase 3 数据文件缺失
   - phase3_requests.pkl
   - phase3_events.pkl
   → 解决方案：Phase 3 不需要额外数据文件！注释掉 load_dataset

2. 环境没有 load_dataset 方法
   → 这是正常的，main.py 已经用 hasattr 检查了

3. 环境没有 request_list
   → 课程学习无法初始化，但不影响训练

4. Phase 3 根本没有运行
   → 检查是否在 Phase 2 之后就停止了

下一步：
1. 查看完整的训练日志
2. 检查是否有错误导致提前退出
3. 确认 Phase 2 完成后是否继续 Phase 3
""")

print("=" * 70)