"""
深度诊断：为什么最多只能连接4/5个目的节点？
分析树构建阶段的具体问题
"""

import sys
sys.path.append('.')

from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
import numpy as np
from collections import defaultdict

def analyze_tree_building_failure():
    print("=" * 80)
    print("树构建失败深度诊断")
    print("=" * 80)

    config = load_config('phase3')
    env = SFC_HIRL_Env(config, use_gnn=True)

    # 运行多个Episode，收集失败模式
    failure_patterns = []

    for ep in range(10):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        req = env.current_request
        src = req.get('source', -1)
        dests = req.get('dest', [])
        vnfs = req.get('vnf', [])

        print(f"\n{'='*80}")
        print(f"Episode {ep}")
        print(f"Source: {src}, Dests: {dests}, VNFs: {vnfs}")
        print('='*80)

        # 先完成VNF部署
        step = 0
        deployed = 0
        current_node = src

        while deployed < len(vnfs) and step < 20:
            step += 1
            mask = env.get_low_level_action_mask()
            valid_actions = np.where(mask)[0]

            deployed_count = len(env.current_tree.get('placement', {}))

            if current_node in valid_actions:
                can_deploy = env._check_deployment_validity(current_node)
                if can_deploy and current_node in env.dc_nodes:
                    env.step_high_level(0)
                    env.step_low_level(current_node)
                    new_deployed = len(env.current_tree.get('placement', {}))
                    if new_deployed > deployed_count:
                        deployed = new_deployed
                        print(f"  ✅ VNF#{deployed-1} 部署在节点{current_node}")
                    continue

            # 移动到DC
            dc_options = [n for n in valid_actions if n in env.dc_nodes]
            if dc_options:
                next_node = dc_options[0]
                env.step_high_level(0)
                env.step_low_level(next_node)
                current_node = env.current_node_location

        if deployed < len(vnfs):
            print(f"  ❌ VNF部署失败: {deployed}/{len(vnfs)}")
            continue

        print(f"  ✅ VNF部署完成: {deployed}/{len(vnfs)}")

        # 树构建阶段诊断
        print(f"\n  【树构建阶段诊断】")

        connected_dests = set()
        tree_edges = []
        step = 0
        max_tree_steps = 50

        # 记录访问历史
        visit_history = defaultdict(int)
        move_history = []
        stuck_count = 0

        while len(connected_dests) < len(dests) and step < max_tree_steps:
            step += 1

            mask = env.get_low_level_action_mask()
            valid_actions = np.where(mask)[0]
            current_node = env.current_node_location

            # 更新已连接的目的节点
            for dest in dests:
                if dest in env.current_tree.get('connected_dests', set()):
                    connected_dests.add(dest)

            unconnected = [d for d in dests if d not in connected_dests]

            if step % 10 == 0 or len(connected_dests) != len(env.current_tree.get('connected_dests', set())):
                print(f"\n  [Step {step}] 当前位置: {current_node}")
                print(f"    已连接: {len(connected_dests)}/{len(dests)} = {list(connected_dests)}")
                print(f"    未连接: {unconnected}")
                print(f"    可用动作: {len(valid_actions)}/28")

            # 检查是否卡住（当前节点就是未连接的目的节点）
            if current_node in unconnected:
                if current_node in valid_actions:
                    print(f"    → 当前节点{current_node}是未连接目的节点，尝试连接")
                    env.step_high_level(0)
                    result = env.step_low_level(current_node)

                    # 检查是否连接成功
                    if current_node in env.current_tree.get('connected_dests', set()):
                        connected_dests.add(current_node)
                        print(f"    ✅ 连接成功！")
                        stuck_count = 0
                    else:
                        print(f"    ❌ 连接失败（在mask中但无法连接）")
                        stuck_count += 1
                else:
                    print(f"    ⚠️  当前节点{current_node}是目的节点但不在mask中！")
                    stuck_count += 1
            else:
                # 选择移动方向
                if len(valid_actions) == 0:
                    print(f"    ❌ 没有可用动作！")
                    break

                # 计算到未连接节点的距离
                best_action = None
                min_dist = float('inf')

                for action in valid_actions:
                    if action == current_node:
                        continue

                    # 计算到最近未连接节点的距离
                    for dest in unconnected:
                        try:
                            path = env.topology_mgr.get_shortest_path(action, dest)
                            if path:
                                dist = len(path) - 1
                                if dist < min_dist:
                                    min_dist = dist
                                    best_action = action
                        except:
                            pass

                if best_action is None:
                    # 随机选择
                    best_action = np.random.choice(valid_actions)

                # 检查是否在循环移动
                visit_history[current_node] += 1
                move_history.append((current_node, best_action))

                if len(move_history) > 10:
                    recent = move_history[-10:]
                    unique_moves = set(recent)
                    if len(unique_moves) <= 3:
                        print(f"    ⚠️  检测到移动循环: {unique_moves}")
                        stuck_count += 1

                # 执行移动
                env.step_high_level(0)
                result = env.step_low_level(best_action)

                if len(result) == 5:
                    _, _, term, trunc, _ = result
                    done = term or trunc
                else:
                    _, _, done, _ = result

                if done:
                    print(f"    ⚠️  Episode提前结束")
                    break

            # 如果卡住太久，退出
            if stuck_count >= 5:
                print(f"\n    ❌ 卡住检测：{stuck_count}次无进展")
                break

        # Episode总结
        print(f"\n  【Episode {ep} 总结】")
        print(f"    最终连接: {len(connected_dests)}/{len(dests)}")
        print(f"    树构建步数: {step}")
        print(f"    总访问次数最多的节点: {max(visit_history.items(), key=lambda x: x[1]) if visit_history else 'N/A'}")

        # 分析失败原因
        if len(connected_dests) < len(dests):
            failure_pattern = {
                'ep': ep,
                'connected': len(connected_dests),
                'total': len(dests),
                'unconnected': unconnected,
                'steps': step,
                'stuck_at': current_node,
                'visit_counts': dict(visit_history),
            }

            # 分析为什么无法连接
            print(f"\n    【失败分析】")

            for dest in unconnected:
                # 检查从当前位置到未连接节点的路径
                try:
                    path = env.topology_mgr.get_shortest_path(current_node, dest)
                    if path:
                        print(f"      到节点{dest}: {len(path)-1}跳, 路径存在")

                        # 检查路径上的节点是否在可用动作中
                        if len(path) > 1:
                            next_hop = path[1]
                            mask = env.get_low_level_action_mask()
                            if next_hop in np.where(mask)[0]:
                                print(f"        下一跳{next_hop}在mask中 ✅")
                            else:
                                print(f"        下一跳{next_hop}不在mask中 ❌")

                                # 分析为什么不在mask中
                                neighbors = env.resource_mgr.get_neighbors(current_node)
                                if next_hop not in neighbors:
                                    print(f"          原因: 不是物理邻居")
                                else:
                                    print(f"          原因: 被mask过滤（可能是访问频次或其他原因）")
                    else:
                        print(f"      到节点{dest}: ❌ 路径不存在")
                except Exception as e:
                    print(f"      到节点{dest}: 错误 {e}")

            failure_patterns.append(failure_pattern)

    # 总结所有失败模式
    print(f"\n{'='*80}")
    print("失败模式总结")
    print('='*80)

    if failure_patterns:
        print(f"\n共{len(failure_patterns)}个Episode失败")

        # 统计连接情况
        connection_stats = defaultdict(int)
        for fp in failure_patterns:
            connection_stats[fp['connected']] += 1

        print(f"\n连接数统计:")
        for connected, count in sorted(connection_stats.items()):
            print(f"  {connected}/5: {count}次")

        # 统计最常见的卡住位置
        stuck_at_counts = defaultdict(int)
        for fp in failure_patterns:
            stuck_at_counts[fp['stuck_at']] += 1

        print(f"\n最常卡住的节点:")
        for node, count in sorted(stuck_at_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  节点{node}: {count}次")

        # 分析根本原因
        print(f"\n【根本原因分析】")
        print(f"1. Mask过于严格")
        print(f"   - 树构建阶段的mask可能排除了通往未连接节点的路径")
        print(f"   - 节点访问频次限制可能太严（visit_count >= 3）")
        print(f"2. 循环移动")
        print(f"   - Agent在少数几个节点间循环移动")
        print(f"   - 没有有效的导航策略")
        print(f"3. 可达性检查失败")
        print(f"   - _find_best_path_to_unconnected可能失败")
        print(f"   - 导致mask中没有可用动作")

        print(f"\n【建议修复】")
        print(f"1. 放宽节点访问频次限制（visit_count < 5）")
        print(f"2. 改进可达性检查逻辑")
        print(f"3. 增加导航提示（朝向未连接节点）")
        print(f"4. 如果mask为空，允许所有物理邻居")
    else:
        print(f"✅ 所有Episode都成功连接所有目的节点！")

if __name__ == "__main__":
    analyze_tree_building_failure()