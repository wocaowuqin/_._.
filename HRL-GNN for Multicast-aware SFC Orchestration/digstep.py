"""
Phase 3 训练诊断工具
检测：步数耗尽 / 死循环 / 资源不足 / 路径问题
"""

import sys
sys.path.append('.')

import logging
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent import GoalConditionedHRLAgent
import numpy as np

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_training_failures():
    print("=" * 80)
    print("Phase 3 训练诊断")
    print("=" * 80)

    config = load_config('phase3')
    env = SFC_HIRL_Env(config, use_gnn=True)

    # 检查环境配置
    print("\n【1. 环境配置】")
    print(f"  Max Steps: {env.max_steps}")
    print(f"  节点数: {env.n}")
    print(f"  DC节点数: {len(env.dc_nodes)}")
    print(f"  DC节点: {env.dc_nodes}")

    # 运行多个测试 Episode
    num_test_episodes = 5
    results = []

    print(f"\n【2. 运行 {num_test_episodes} 个测试 Episode】")
    print("-" * 80)

    for ep in range(num_test_episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        if env.current_request is None:
            continue

        req = env.current_request
        src = req.get('source', -1)
        dests = req.get('dest', [])
        vnf_chain = req.get('vnf', [])

        print(f"\n测试 Episode {ep}:")
        print(f"  请求: Src={src}, Dests={dests}, VNFs={vnf_chain}")

        # 追踪信息
        step_count = 0
        phase = 'vnf'
        visited_nodes = []
        connected_dests = set()
        node_visit_frequency = {}
        stuck_detected = False
        resource_failed = False
        last_10_nodes = []

        done = False
        total_reward = 0

        while not done and step_count < env.max_steps + 10:
            step_count += 1

            # 记录访问的节点
            current_node = env.current_node_location
            visited_nodes.append(current_node)
            last_10_nodes.append(current_node)
            if len(last_10_nodes) > 10:
                last_10_nodes.pop(0)

            node_visit_frequency[current_node] = node_visit_frequency.get(current_node, 0) + 1

            # 检测死循环：最近10步都在同一个节点
            if len(last_10_nodes) == 10 and len(set(last_10_nodes)) <= 2:
                stuck_detected = True
                print(f"  ⚠️  [Step {step_count}] 检测到死循环！最近10步: {last_10_nodes}")

            # 获取 mask
            high_mask = env.get_high_level_action_mask()
            low_mask = env.get_low_level_action_mask()

            # 检查是否有可用动作
            valid_actions = np.sum(low_mask)
            if valid_actions == 0:
                print(f"  ❌ [Step {step_count}] 没有可用动作！当前节点: {current_node}")
                break

            # 随机选择动作（简化，不用 Agent）
            valid_indices = np.where(low_mask)[0]
            action = np.random.choice(valid_indices)

            # 执行动作
            env.step_high_level(0)  # 假设高层动作
            step_result = env.step_low_level(action)

            if len(step_result) == 5:
                next_state, r, term, trunc, info = step_result
                done = term or trunc
            else:
                next_state, r, done, info = step_result

            total_reward += r

            # 检测阶段变化
            if info.get('phase') == 'tree_building' and phase == 'vnf':
                phase = 'tree'
                print(f"  ✅ [Step {step_count}] VNF部署完成，进入树构建阶段")

            # 检测连接进度
            if info.get('dest_connected'):
                connected_dests.add(env.current_node_location)
                print(f"  🎯 [Step {step_count}] 连接目标 {env.current_node_location} ({len(connected_dests)}/{len(dests)})")

            # 检测资源失败
            if info.get('error') in ['deploy_failed', 'no_resource']:
                resource_failed = True
                print(f"  ❌ [Step {step_count}] 资源不足：{info.get('error')}")

            state = next_state

        # Episode 结束分析
        print(f"\n  结果分析:")
        print(f"    总步数: {step_count}")
        print(f"    总奖励: {total_reward:.2f}")
        print(f"    最终阶段: {phase}")
        print(f"    连接进度: {len(connected_dests)}/{len(dests)}")

        # 失败原因判断
        failure_reason = []

        if step_count >= env.max_steps:
            failure_reason.append("步数耗尽")

        if stuck_detected:
            failure_reason.append("死循环")

        if resource_failed:
            failure_reason.append("资源不足")

        if len(connected_dests) < len(dests):
            failure_reason.append(f"未完成连接 ({len(connected_dests)}/{len(dests)})")

        # 节点访问频率分析
        max_visit = max(node_visit_frequency.values()) if node_visit_frequency else 0
        most_visited = [n for n, c in node_visit_frequency.items() if c == max_visit]

        if max_visit > 20:
            failure_reason.append(f"过度访问节点{most_visited} ({max_visit}次)")

        print(f"    失败原因: {', '.join(failure_reason) if failure_reason else '成功'}")
        print(f"    访问最多的节点: {most_visited[0] if most_visited else 'N/A'} ({max_visit}次)")

        results.append({
            'episode': ep,
            'steps': step_count,
            'reward': total_reward,
            'connected': len(connected_dests),
            'total_dests': len(dests),
            'phase': phase,
            'stuck': stuck_detected,
            'resource_fail': resource_failed,
            'max_visit': max_visit,
            'failure_reasons': failure_reason
        })

        print("-" * 80)

    # 总结分析
    print("\n【3. 统计分析】")
    print("=" * 80)

    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_connected = np.mean([r['connected'] for r in results])
    avg_total = np.mean([r['total_dests'] for r in results])

    print(f"平均步数: {avg_steps:.1f} / {env.max_steps}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均连接: {avg_connected:.1f} / {avg_total:.1f}")

    # 失败原因统计
    all_reasons = []
    for r in results:
        all_reasons.extend(r['failure_reasons'])

    from collections import Counter
    reason_counts = Counter(all_reasons)

    print(f"\n失败原因统计:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")

    # 给出建议
    print("\n【4. 优化建议】")
    print("=" * 80)

    if "步数耗尽" in reason_counts and reason_counts["步数耗尽"] >= len(results) * 0.5:
        print("✅ 主要问题：步数耗尽")
        print("   建议：增加 max_steps 到 200-300")
        print(f"   当前 max_steps: {env.max_steps}")

    if "死循环" in reason_counts:
        print("✅ 检测到死循环")
        print("   建议：增强频次惩罚或改进路径引导")

    if "资源不足" in reason_counts:
        print("✅ 存在资源不足")
        print("   建议：检查资源管理器的资源分配")

    stuck_count = sum(1 for r in results if r['stuck'])
    if stuck_count > 0:
        print(f"✅ {stuck_count}/{len(results)} 个Episode出现卡住")
        print("   建议：检查动作选择逻辑")

    if avg_connected < avg_total * 0.5:
        print("✅ 连接率过低")
        print("   建议：检查路径规划和奖励设计")

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_training_failures()