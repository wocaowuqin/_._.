"""
main.py - 完整修复版 + 训练诊断

主要修复:
1. ✅ 加载拓扑矩阵
2. ✅ 使用统一的 Agent
3. ✅ 修复配置键名
4. ✅ 安全的配置访问
5. 🔥 新增训练前诊断
"""
import scipy.io
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent.agent import HRL_DQN_Agent
from trainer.phase1_collector import Phase1ExpertCollector
from trainer.phase2_il_trainer import Phase2ILTrainer
from trainer.phase3_rl_trainer import Phase3RLTrainer

# 配置全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_config_path(config, key_path):
    """安全地获取配置路径"""
    possible_locations = [
        ['path', key_path],
        ['project', key_path],
        ['paths', key_path],
        [key_path],
    ]

    for location in possible_locations:
        try:
            value = config
            for key in location:
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue

    default_paths = {
        'ckpt_dir': './outputs/checkpoints',
        'log_dir': './outputs/logs',
        'expert_data_dir': './data/expert',
        'input_dir': './data/input_dir',
    }

    if key_path in default_paths:
        logger.warning(f"⚠️  配置中未找到 {key_path}，使用默认值: {default_paths[key_path]}")
        return default_paths[key_path]

    return None


def ensure_paths_exist(config):
    """确保所有必要的目录存在"""
    path_keys = ['ckpt_dir', 'log_dir', 'expert_data_dir']

    for key in path_keys:
        path = get_config_path(config, key)
        if path:
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ 目录准备完成: {path}")


def validate_config(config, phase):
    """验证配置完整性"""
    logger.info("🔍 验证配置...")

    errors = []
    warnings = []

    if 'gnn' not in config:
        warnings.append("⚠️  缺少 'gnn' 配置块（将从环境动态获取）")

    if 'env' not in config and 'environment' not in config:
        errors.append("❌ 缺少 'env' 或 'environment' 配置块")

    if phase not in config:
        warnings.append(f"⚠️  缺少 '{phase}' 配置块")

    if errors:
        logger.error("❌ 配置验证失败:")
        for err in errors:
            logger.error(f"  {err}")
        raise ValueError("配置不完整，请检查配置文件")

    if warnings:
        logger.warning("⚠️  配置警告:")
        for warn in warnings:
            logger.warning(f"  {warn}")

    logger.info("✅ 配置验证通过")


def load_topology(config):
    """
    Unified topology loader (NO FALLBACK).
    """
    logger.info("📡 正在加载拓扑矩阵...")

    if 'topology' not in config:
        config['topology'] = {}

    try:
        input_dir = config['path']['input_dir']
    except KeyError:
        logger.error("❌ 缺少 config['path']['input_dir']")
        return False

    mat_path = os.path.join(input_dir, 'US_Backbone_path.mat')

    if not os.path.exists(mat_path):
        logger.error(f"❌ 拓扑文件不存在: {mat_path}")
        return False

    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        logger.error(f"❌ 读取 mat 文件失败: {e}")
        return False

    for key, val in mat_data.items():
        if key.startswith('__'):
            continue

        if isinstance(val, np.ndarray) and val.ndim == 2:
            if val.shape[0] == val.shape[1] and np.issubdtype(val.dtype, np.number):
                if np.sum(val) > val.shape[0]:
                    logger.info(f"✅ 使用邻接矩阵字段: {key}")
                    topo = (val > 0).astype(np.float32)
                    np.fill_diagonal(topo, 0)

                    if np.sum(topo) == 0:
                        logger.error("❌ 邻接矩阵为空")
                        return False

                    config['topology']['matrix'] = topo
                    return True

    if 'Paths' not in mat_data:
        logger.error("❌ mat 文件中不存在 Paths 结构")
        return False

    logger.info("🔍 解析 Paths 构建拓扑...")
    paths_matrix = mat_data['Paths']
    N, M = paths_matrix.shape

    topo = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            if i == j:
                continue

            cell = paths_matrix[i, j]

            if not hasattr(cell, 'dtype'):
                continue
            if cell.dtype.names is None:
                continue
            if 'paths' not in cell.dtype.names:
                continue

            paths_array = cell['paths']
            if not isinstance(paths_array, np.ndarray):
                continue

            if paths_array.ndim == 1:
                paths_array = paths_array[np.newaxis, :]

            for path in paths_array:
                nodes = path[path > 0] - 1
                if len(nodes) < 2:
                    continue

                for k in range(len(nodes) - 1):
                    u, v = int(nodes[k]), int(nodes[k + 1])
                    if 0 <= u < N and 0 <= v < N:
                        topo[u, v] = 1.0
                        topo[v, u] = 1.0

    np.fill_diagonal(topo, 0)

    if np.sum(topo) == 0:
        logger.error("❌ Paths 解析完成，但未发现任何物理链路")
        return False

    num_edges = int(np.sum(topo) / 2)
    logger.info(f"✅ 拓扑加载成功: {N} 节点, {num_edges} 条物理链路")

    config['topology']['matrix'] = topo.astype(np.float32)
    return True


def inject_dynamic_dimensions(config, env):
    """从环境中获取动态维度并注入到配置"""
    logger.info("🔧 注入动态维度...")

    if 'gnn' not in config:
        config['gnn'] = {}

    config['gnn']['node_feat_dim'] = env.resource_mgr.node_feat_dim
    config['gnn']['edge_feat_dim'] = env.resource_mgr.edge_feat_dim
    config['gnn']['request_feat_dim'] = env.resource_mgr.request_dim

    logger.info(f"  node_feat_dim: {config['gnn']['node_feat_dim']}")
    logger.info(f"  edge_feat_dim: {config['gnn']['edge_feat_dim']}")
    logger.info(f"  request_feat_dim: {config['gnn']['request_feat_dim']}")


def main():
    parser = argparse.ArgumentParser(description="HRL-GNN SFC Orchestration Training Pipeline")
    parser.add_argument('--phase', type=str, required=True,
                        choices=['phase1', 'phase2', 'phase3'],
                        help='Training phase')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"🖥️  使用 GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        logger.info("🖥️  使用 CPU")

    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"🌱 随机种子: {args.seed}")

    # 加载配置
    try:
        logger.info(f"📂 加载配置: {args.phase}")
        config = load_config(args.phase)

        if 'eval' not in config:
            config['eval'] = {}
        config['eval']['device'] = str(device)
        config['eval']['seed'] = args.seed

        logger.info(f"✅ 配置加载成功")
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 验证配置
    try:
        validate_config(config, args.phase)
    except ValueError as e:
        logger.error(str(e))
        return

    # 确保目录存在
    ensure_paths_exist(config)

    # 加载拓扑矩阵
    if not load_topology(config):
        logger.error("❌ 拓扑矩阵加载失败，无法继续")
        return

    # Phase 1: 专家数据采集
    if args.phase == 'phase1':
        logger.info("=" * 70)
        logger.info("🚀 Phase 1: Expert Data Collection")
        logger.info("=" * 70)

        try:
            env = SFC_HIRL_Env(config, use_gnn=True)
            logger.info("✅ 环境初始化成功")
        except Exception as e:
            logger.error(f"❌ 环境初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            expert_solver = env.policy_helper.expert
            logger.info("✅ Expert Solver 已加载")
        except Exception as e:
            logger.error(f"❌ Expert Solver 获取失败: {e}")
            return

        output_dir = get_config_path(config, 'expert_data_dir')
        max_episodes = config.get("phase1", {}).get("max_episodes", 5000)
        save_every = config.get("phase1", {}).get("save_every", 500)

        collector = Phase1ExpertCollector(
            env=env,
            expert_solver=expert_solver,
            output_dir=output_dir,
            max_episodes=max_episodes,
            save_every=save_every,
        )

        try:
            collector.collect()
            logger.info("✅ Phase 1 完成")
        except Exception as e:
            logger.error(f"❌ Phase 1 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # Phase 2: 模仿学习
    elif args.phase == 'phase2':
        logger.info("=" * 70)
        logger.info("🚀 Phase 2: Imitation Learning")
        logger.info("=" * 70)

        try:
            temp_env = SFC_HIRL_Env(config, use_gnn=True)
            inject_dynamic_dimensions(config, temp_env)
            del temp_env
            logger.info("✅ 动态维度注入成功")
        except Exception as e:
            logger.error(f"❌ 环境初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            logger.info("🔧 初始化 Phase 2 Agent...")
            agent = HRL_DQN_Agent(config, phase=2)
            logger.info("✅ Agent 初始化成功")
            logger.info(f"   模式: Phase {agent.phase}")
            logger.info(f"   动作空间: {agent.n_actions}")
            logger.info(f"   设备: {agent.device}")

            if hasattr(agent, 'policy_net'):
                logger.info("   ✅ policy_net 已加载")
            else:
                logger.error("   ❌ policy_net 未找到")
                return

        except Exception as e:
            logger.error(f"❌ Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        data_file = "expert_data_final.pkl"
        expert_data_dir = get_config_path(config, 'expert_data_dir')
        data_path = os.path.join(expert_data_dir, data_file)

        if not os.path.exists(data_path):
            logger.error(f"❌ 专家数据不存在: {data_path}")
            logger.error("   请先运行 Phase 1 收集数据")
            return

        phase2_config = config.get('phase2', {})
        output_dir = get_config_path(config, 'ckpt_dir')

        try:
            trainer = Phase2ILTrainer(
                agent=agent,
                expert_data_path=data_path,
                output_dir=output_dir,
                config=phase2_config
            )
            logger.info("✅ Phase2 Trainer 初始化成功")
        except Exception as e:
            logger.error(f"❌ Trainer 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            trainer.run()
            logger.info("✅ Phase 2 完成")
        except Exception as e:
            logger.error(f"❌ Phase 2 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # Phase 3: 强化学习微调
    elif args.phase == 'phase3':
        logger.info("=" * 70)
        logger.info("🚀 Phase 3: RL Fine-tuning")
        logger.info("=" * 70)

        # 初始化环境
        try:
            env = SFC_HIRL_Env(config, use_gnn=True)
            inject_dynamic_dimensions(config, env)
            logger.info("✅ 环境初始化成功")
        except Exception as e:
            logger.error(f"❌ 环境初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 创建 Agent
        try:
            logger.info("🔧 初始化 Agent...")

            agent = HRL_DQN_Agent(
                config,
                high_action_dim=env.NB_HIGH_LEVEL_GOALS,
                low_action_dim=env.NB_LOW_LEVEL_ACTIONS,
                state_dim=env.observation_space['x'].shape[1] if hasattr(env, 'observation_space') else None
            )

            logger.info("✅ Agent 初始化成功")
            logger.info(f"   动作空间: High={env.NB_HIGH_LEVEL_GOALS}, Low={env.NB_LOW_LEVEL_ACTIONS}")
            logger.info(f"   设备: {agent.device}")

        except Exception as e:
            logger.error(f"❌ Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 加载预训练模型
        ckpt_dir = get_config_path(config, 'ckpt_dir')
        pretrained_path = os.path.join(ckpt_dir, "il_model_final.pth")

        if os.path.exists(pretrained_path):
            logger.info(f"📥 加载预训练模型: {pretrained_path}")
            try:
                agent.load(pretrained_path)
                logger.info("✅ 预训练模型加载成功")
            except Exception as e:
                logger.warning(f"⚠️  预训练模型加载失败: {e}")
                logger.warning("   将从随机初始化开始训练")
        else:
            logger.warning(f"⚠️  未找到预训练模型: {pretrained_path}")
            logger.warning("   将从随机初始化开始训练")

        # 创建 Trainer
        trainer = Phase3RLTrainer(
            env=env,
            agent=agent,
            output_dir=ckpt_dir,
            config=config
        )

        # ============================================================
        # 🔥 训练前诊断（排查 Epsilon 问题）
        # ============================================================
        print("=" * 60)
        print("🔬 训练前诊断:")
        print(f"  Config 路径: config/phase3.yaml")
        print(f"  Phase3 配置存在: {'phase3' in config}")

        # 检查 Epsilon 配置
        eps_cfg = config.get('phase3', {}).get('epsilon', {})
        print(f"  Epsilon Initial: {eps_cfg.get('initial', 'NOT FOUND')}")
        print(f"  Epsilon Final: {eps_cfg.get('final', 'NOT FOUND')}")
        print(f"  Epsilon Decay Steps: {eps_cfg.get('decay_steps', 'NOT FOUND')}")

        # 检查 RL 配置
        rl_cfg = config.get('phase3', {}).get('rl', {})
        print(f"  Learning Rate: {rl_cfg.get('learning_rate', 'NOT FOUND')}")
        print(f"  Batch Size: {rl_cfg.get('replay_buffer', {}).get('batch_size', 'NOT FOUND')}")
        print(f"  Min Buffer Size: {rl_cfg.get('replay_buffer', {}).get('min_size', 'NOT FOUND')}")

        # 检查 Trainer 内部参数
        print(f"\n  Trainer 内部参数:")
        print(f"    epsilon_initial: {trainer.epsilon_initial}")
        print(f"    epsilon_final: {trainer.epsilon_final}")
        print(f"    epsilon_decay_steps: {trainer.epsilon_decay_steps}")
        print(f"    min_buffer_size: {trainer.min_buffer_size}")

        # 检查 Agent
        print(f"\n  Agent 参数:")
        print(f"    Epsilon: {trainer.agent.epsilon if hasattr(trainer.agent, 'epsilon') else 'NO EPSILON'}")
        print(
            f"    LR: {trainer.agent.optimizer.param_groups[0]['lr'] if hasattr(trainer.agent, 'optimizer') else 'NO OPTIMIZER'}")

        # 预测未来的 Epsilon
        print(f"\n  预期 Epsilon 衰减:")
        for ep in [0, 10, 50, 100, 500]:
            steps = ep * 120
            eps = trainer._calculate_epsilon(steps)
            print(f"    Episode {ep:3d} (Step {steps:5d}): {eps:.4f}")

        print("=" * 60)
        # ============================================================

        # 开始训练
        try:
            trainer.run()
            logger.info("✅ Phase 3 完成")
        except Exception as e:
            logger.error(f"❌ Phase 3 执行失败: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 70)
    logger.info("🎉 程序执行完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()