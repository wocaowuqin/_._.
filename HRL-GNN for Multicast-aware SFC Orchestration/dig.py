#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent兼容性验证脚本
快速检查 agent.py 是否包含所有必要的兼容性方法
"""

import sys
import torch

print("=" * 60)
print("Agent 兼容性验证")
print("=" * 60)

# 1. 导入测试
print("\n1️⃣ 测试导入...")
try:
    from core.hrl.agent import HRLAgent, GoalConditionedHRLAgent

    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 2. 创建配置
print("\n2️⃣ 创建配置...")
config = {
    'hrl': {
        'state_dim': 128,
        'goal_dim': 64,
        'hidden_dim': 128,
        'subgoal_horizon': 20,
        'intrinsic_reward_weight': 0.3
    },
    'environment': {
        'nb_low_level_actions': 50,
        'nb_high_level_goals': 10
    },
    'training': {
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'buffer_size': 10000,
        'target_update_freq': 1000,
        'epsilon': {
            'initial': 0.3,
            'final': 0.05,
            'decay_steps': 50000
        }
    },
    'use_cuda': False,
    'dropout': 0.1
}
print("✅ 配置创建成功")

# 3. 创建Agent (新方式)
print("\n3️⃣ 创建HRLAgent（新方式）...")
try:
    agent_new = HRLAgent(config, encoder=None, phase=3, goal_strategy='adaptive')
    print("✅ HRLAgent创建成功")
except Exception as e:
    print(f"❌ HRLAgent创建失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 4. 创建Agent (旧方式)
print("\n4️⃣ 创建GoalConditionedHRLAgent（旧方式）...")
try:
    agent_old = GoalConditionedHRLAgent(config, phase=3, goal_strategy='adaptive')
    print("✅ GoalConditionedHRLAgent创建成功（向后兼容）")
except Exception as e:
    print(f"❌ GoalConditionedHRLAgent创建失败: {e}")
    sys.exit(1)

# 5. 检查必要属性和方法
print("\n5️⃣ 检查兼容性属性和方法...")

all_passed = True

compatibility_checks = [
    ('_generate_and_encode_subgoal', '方法', True),
    ('_generate_goal_embedding', '方法', True),
    ('goal_embedding', '属性', True),
    ('subgoal_step_count', '属性', False),
    ('subgoal_steps', '属性', False),
    ('current_subgoal', '属性', False),
    ('current_subgoal_emb', '属性', False),
    ('current_goal_emb', '属性', False),
    ('high_policy', '属性', False),
    ('low_policy', '属性', False),
    ('select_action', '方法', True),
    ('store_transition_high', '方法', True),
    ('store_transition_low', '方法', True),
    ('update_policies', '方法', True),
]

for name, type_, is_method in compatibility_checks:
    has_attr = hasattr(agent_new, name)

    if is_method:
        is_callable_ = callable(getattr(agent_new, name, None))
        status = "✅" if has_attr and is_callable_ else "❌"
        if not (has_attr and is_callable_):
            all_passed = False
    else:
        status = "✅" if has_attr else "❌"
        if not has_attr:
            all_passed = False

    print(f"  {status} {name} ({type_})")

if not all_passed:
    print("\n❌ 兼容性检查失败！")
    print("   请确保使用最新的 agent.py")
    sys.exit(1)

print("\n✅ 所有兼容性检查通过")

# 6. 测试 _generate_and_encode_subgoal 方法
print("\n6️⃣ 测试 _generate_and_encode_subgoal 方法...")

try:
    # 创建模拟状态
    state = {
        'current_position': 0,
        'unconnected_dests': [1, 3, 5],
        'step_counter': 0
    }

    # 调用方法
    agent_new._generate_and_encode_subgoal(state)

    # 检查结果 - 注意：这个方法只设置 embedding，不设置整数ID
    assert agent_new.current_subgoal_emb is not None, "current_subgoal_emb 未设置"
    assert agent_new.current_goal_emb is not None, "current_goal_emb 未设置"
    assert agent_new.subgoal_step_count == 0, "subgoal_step_count 未正确初始化"

    print(f"✅ Subgoal embedding生成成功")
    print(f"   - current_subgoal_emb shape: {agent_new.current_subgoal_emb.shape}")
    print(f"   - current_goal_emb shape: {agent_new.current_goal_emb.shape}")
    print(f"   - subgoal_step_count: {agent_new.subgoal_step_count}")
    print(f"   - current_subgoal (节点ID): {agent_new.current_subgoal} (None是正常的)")

except Exception as e:
    print(f"❌ Subgoal生成失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 7. 测试动作选择
print("\n7️⃣ 测试动作选择...")

try:
    import numpy as np

    state = {
        'current_position': 0,
        'unconnected_dests': [1, 3, 5, 7, 9],
        'step_counter': 0
    }
    action_mask = np.ones(50)

    high_action, low_action, info = agent_new.select_action(
        state,
        unconnected_dests=[1, 3, 5, 7, 9],
        action_mask=action_mask,
        use_expert=False
    )

    print(f"✅ 动作选择成功")
    print(f"   - High action: {high_action}")
    print(f"   - Low action: {low_action}")
    print(f"   - High-Level decision: {info['high_level_decision']}")
    print(f"   - Current subgoal: {info['subgoal']} (现在是整数)")

except Exception as e:
    print(f"❌ 动作选择失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 8. 测试经验存储
print("\n8️⃣ 测试经验存储...")

try:
    # Low-Level经验
    agent_new.store_transition_low(state, low_action, 1.0, state, False)

    # High-Level经验
    agent_new.store_transition_high(state, 1, 10.0, state, False)

    print(f"✅ 经验存储成功")
    print(f"   - Low memory size: {len(agent_new.low_memory)}")
    print(f"   - High memory size: {len(agent_new.high_memory)}")

except Exception as e:
    print(f"❌ 经验存储失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 9. 测试goal_embedding属性
print("\n9️⃣ 测试 goal_embedding 属性...")

# 测试所有三种策略
strategies = ['adaptive', 'hybrid', 'relative']

for strategy in strategies:
    print(f"\n   测试 {strategy} 策略...")

    try:
        # 创建agent
        test_agent = HRLAgent(config, encoder=None, phase=3, goal_strategy=strategy)

        assert hasattr(test_agent, 'goal_embedding'), f"缺少 goal_embedding 属性 ({strategy})"
        assert test_agent.goal_embedding is not None, f"goal_embedding 未初始化 ({strategy})"

        # 测试goal_embedding的forward
        graph_emb = torch.randn(1, 128)

        if strategy == 'adaptive':
            complexity = torch.tensor([[0.5]])
            subgoal, info = test_agent.goal_embedding(graph_emb, complexity)
            # AdaptiveSubgoalEmbedding 返回 (subgoal, info)
            # info 是字典，不是 tensor
            print(f"   ✅ {strategy}: Subgoal shape: {subgoal.shape}, Info keys: {list(info.keys())}")

        elif strategy == 'hybrid':
            subgoal, goal_emb, history = test_agent.goal_embedding(graph_emb, return_refinement_history=False)
            print(f"   ✅ {strategy}: Subgoal shape: {subgoal.shape}, Goal emb shape: {goal_emb.shape}")

        else:  # 'relative'
            target_emb = torch.randn_like(graph_emb)
            subgoal, goal_emb = test_agent.goal_embedding(graph_emb, target_emb)
            print(f"   ✅ {strategy}: Subgoal shape: {subgoal.shape}, Goal emb shape: {goal_emb.shape}")

    except Exception as e:
        print(f"   ❌ {strategy} 策略失败: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

if all_passed:
    print(f"\n✅ 所有goal_embedding策略测试通过")
else:
    print(f"\n❌ 部分goal_embedding策略测试失败")
    sys.exit(1)

# 完成
print("\n" + "=" * 60)
print("🎉 所有兼容性测试通过！")
print("=" * 60)
print("\n📋 验证结果:")
print("  ✅ 新旧Agent都可以创建")
print("  ✅ _generate_and_encode_subgoal 方法可用（生成embedding）")
print("  ✅ goal_embedding 属性可用")
print("  ✅ current_subgoal (int) 和 current_subgoal_emb (Tensor) 正确分离")
print("  ✅ 动作选择正常（current_subgoal设置为整数）")
print("  ✅ 经验存储正常")
print("  ✅ 所有goal_embedding策略测试通过")
print("\n🚀 你的 main.py 现在应该可以正常运行了！")
print("\n📚 下一步:")
print("  1. 运行 main.py")
print("  2. 检查训练日志")
print("  3. 监控High/Low决策")