import logging
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent import GoalConditionedHRLAgent, create_goal_conditioned_agent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Diagnose")

def setup_env_and_agent(config_name='phase3'):
    """初始化环境和Agent"""
    logger.info(f"🔧 正在初始化环境 (Config: {config_name})...")
    try:
        config = load_config(config_name)
        # 强制使用 CPU 进行诊断，方便调试
        config['use_cuda'] = False
        config['device'] = 'cpu'

        # 初始化环境
        env = SFC_HIRL_Env(config)

        # 获取Encoder
        encoder = None
        if hasattr(env, 'resource_mgr') and hasattr(env.resource_mgr, 'encoder'):
            encoder = env.resource_mgr.encoder

        # 初始化Agent
        # 注意：这里我们加载 Phase 3 的配置，但用于推理
        agent = create_goal_conditioned_agent(config, phase=3, encoder=encoder)

        # 尝试加载模型权重 (如果有的话)
        ckpt_path = "outputs/checkpoints/rl_model_final.pth"
        if os.path.exists(ckpt_path):
            logger.info(f"📥 加载模型权重: {ckpt_path}")
            agent.load(ckpt_path)
        else:
            logger.warning("⚠️ 未找到训练好的模型，Agent 将使用随机策略！")

        agent.eval() # 切换到评估模式

        return env, agent
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        sys.exit(1)

def run_diagnostic_episode(env, agent, mode='agent'):
    """运行单个诊断回合"""
    state, info = env.reset()
    if env.current_request is None:
        return None

    done = False
    steps = 0
    total_reward = 0

    # 诊断计数器
    mask_blocks = 0  # 被Mask拦截的次数
    loops = 0        # 发生的死循环次数

    visited_nodes = []

    while not done:
        # 获取 Mask
        high_mask = env.get_high_level_action_mask()
        low_mask = env.get_low_level_action_mask()
        curr_node = env.current_node_location
        visited_nodes.append(curr_node)

        action = None

        # ====================================================
        # 模式 A: 纯专家模式 (诊断专家是否靠谱)
        # ====================================================
        if mode == 'expert':
            # 获取专家建议
            try:
                # 模拟 trainer 中的专家调用逻辑
                # 这里简化处理：直接问环境有没有专家实例
                if hasattr(env, 'expert') and env.expert:
                    req = env.current_request
                    # 简单启发式：去最近的目的地
                    dests = req.get('dest', [])
                    path, _ = env.expert.find_any_path(curr_node, dests[0] if dests else 0)
                    if path and len(path) > 1:
                        expert_act = path[1]

                        # 🔥 关键检查：专家选的路，环境允许吗？
                        if low_mask[expert_act] == 1:
                            action = expert_act
                        else:
                            # 专家撞墙了！
                            mask_blocks += 1
                            # logger.warning(f"   ⚠️ 专家试图走非法路径: {curr_node}->{expert_act} (Masked)")
                            # 这种情况下随机选一个合法的走，让游戏继续
                            valid = np.where(low_mask)[0]
                            action = np.random.choice(valid) if len(valid) > 0 else 0
            except:
                pass

            if action is None:
                # 如果专家没算出路，随机走
                valid = np.where(low_mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else 0

        # ====================================================
        # 模式 B: 纯 Agent 模式 (诊断 Agent 学没学会)
        # ====================================================
        elif mode == 'agent':
            # 使用 Agent 决策 (Epsilon=0)
            # 注意：传入正确的参数
            high, low, _ = agent.select_action(
                state,
                action_mask=low_mask,
                epsilon=0.0 # 贪婪模式
            )
            action = low

        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

        # 检查死循环 (最近5步是否有重复)
        if len(visited_nodes) > 5:
            recent = visited_nodes[-5:]
            if len(set(recent)) < 3: # 简单判定：5步内只访问了不到3个不同节点
                loops += 1

        if done:
            break

    return {
        "success": info.get('request_completed', False),
        "reward": total_reward,
        "steps": steps,
        "mask_blocks": mask_blocks,
        "loops": loops
    }

def main():
    env, agent = setup_env_and_agent()

    # 准备测试集 (取前20个请求)
    num_tests = 20
    logger.info(f"\n🚀 开始诊断 (测试请求数: {num_tests})")

    # ============================================================
    # 🧪 测试 1: 专家可靠性测试
    # ============================================================
    logger.info("\n🧪 [测试 1] 纯专家模式 (Pure Expert)")
    logger.info("   目的: 检查专家算法是否与环境约束(Mask)冲突")

    expert_stats = {'success': 0, 'blocks': 0}
    env.data_loader.reset() # 重置数据流

    for _ in tqdm(range(num_tests), desc="Expert Test"):
        res = run_diagnostic_episode(env, agent, mode='expert')
        if res:
            if res['success']: expert_stats['success'] += 1
            expert_stats['blocks'] += res['mask_blocks']

    expert_acc = (expert_stats['success'] / num_tests) * 100
    avg_blocks = expert_stats['blocks'] / num_tests

    logger.info(f"   👉 专家成功率: {expert_acc:.1f}%")
    logger.info(f"   👉 专家违规次数/Ep: {avg_blocks:.2f}")
    if avg_blocks > 0.5:
        logger.error("   ❌ 诊断结论: 专家策略有严重问题！它经常建议走非法路径(Masked)。")
        logger.error("      建议: 修改 Trainer 代码，在采纳专家建议前必须检查 Mask。")
    else:
        logger.info("   ✅ 诊断结论: 专家策略基本健康。")

    # ============================================================
    # 🧪 测试 2: Agent 学习情况测试
    # ============================================================
    logger.info("\n🧪 [测试 2] 纯 Agent 模式 (Pure Agent)")
    logger.info("   目的: 检查模型是否学会了基本的导航")

    agent_stats = {'success': 0, 'reward': [], 'loops': 0}
    env.data_loader.reset()

    for _ in tqdm(range(num_tests), desc="Agent Test"):
        res = run_diagnostic_episode(env, agent, mode='agent')
        if res:
            if res['success']: agent_stats['success'] += 1
            agent_stats['reward'].append(res['reward'])
            agent_stats['loops'] += res['loops']

    agent_acc = (agent_stats['success'] / num_tests) * 100
    avg_rw = np.mean(agent_stats['reward']) if agent_stats['reward'] else 0

    logger.info(f"   👉 Agent 成功率: {agent_acc:.1f}%")
    logger.info(f"   👉 平均奖励: {avg_rw:.2f}")
    logger.info(f"   👉 死循环迹象: {agent_stats['loops']} 次")

    if agent_acc < 10.0:
        logger.warning("   ⚠️ 诊断结论: Agent 尚未学会有效策略 (成功率低)。")
        if avg_rw < -50:
             logger.info("      观察: 奖励非常低，说明惩罚机制生效了，Agent 应该正在吸取教训。")
        else:
             logger.warning("      观察: 奖励绝对值较小，可能 Reward Scale 太小，Agent 感觉不到痛。")
    else:
        logger.info("   ✅ 诊断结论: Agent 表现良好，已具备一定能力。")

    # ============================================================
    # 🧪 综合建议
    # ============================================================
    logger.info("\n📋 最终处方:")
    if expert_stats['success'] < 50:
        print("1. [紧急] 您的专家算法不可靠。在 RL 训练中，请务必过滤掉非法的专家动作。")
    if agent_acc < 5:
        print("2. [正常] Agent 还在初期阶段。只要 Expert 修复了，Agent 会慢慢跟上的。")
    if avg_rw > -10 and agent_acc < 5:
        print("3. [建议] 检查 Reward 函数。如果任务失败了奖励却没怎么扣(例如总分 -2)，Agent 学得会很慢。失败惩罚应该大一点(例如 -100)。")

if __name__ == "__main__":
    main()