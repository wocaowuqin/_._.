import argparse
import logging
import os
import sys
import numpy as np
import torch
import random

# 添加项目根目录到 Python 路径，防止找不到包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent import HRL_DQN_Agent  # 需确保 core/hrl/agent.py 已创建
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


def main():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="HRL-GNN SFC Orchestration Training Pipeline")
    parser.add_argument('--phase', type=str, required=True, choices=['phase1', 'phase2', 'phase3'],
                        help='Training phase: phase1(Expert), phase2(IL), phase3(RL)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (if available)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # ==========================================
    # 2. 基础设置 (Device & Config)
    # ==========================================
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    set_seed(args.seed)
    logger.info(f"🚀 Running {args.phase.upper()} on {device}")

    try:
        # 自动加载 base.yaml + model.yaml + phaseX.yaml
        config = load_config(args.phase)

        # 强制覆盖配置中的 device 和 seed
        if 'eval' not in config: config['eval'] = {}
        config['eval']['device'] = str(device)
        config['eval']['seed'] = args.seed

        logger.info(f"✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    # 确保关键目录存在
    os.makedirs(config['path']['ckpt_dir'], exist_ok=True)
    os.makedirs(config['path']['log_dir'], exist_ok=True)
    os.makedirs(config['path']['expert_data_dir'], exist_ok=True)

    # ==========================================
    # 3. 分阶段执行逻辑
    # ==========================================

    # ------------------------------------
    # Phase 1: 专家数据采集 (Expert Collection)
    # ------------------------------------
    if args.phase == 'phase1':
        logger.info(">>> Initializing Phase 1: Expert Data Collection")

        # 初始化环境 (开启 GNN 模式以统一数据格式)
        env = SFC_HIRL_Env(config, use_gnn=True)

        collector = Phase1ExpertCollector(
            env=env,
            output_dir=config['path']['expert_data_dir'],
            config=config.get('expert', {})  # 对应 phase1.yaml 中的 expert 块
        )
        collector.run()

    # ------------------------------------
    # Phase 2: 模仿学习 (Imitation Learning)
    # ------------------------------------
    elif args.phase == 'phase2':
        logger.info(">>> Initializing Phase 2: Imitation Learning")

        # 1. 临时初始化环境以获取网络维度 (Input Dims)
        temp_env = SFC_HIRL_Env(config, use_gnn=True)
        config['gnn']['node_feat_dim'] = temp_env.resource_mgr.node_feat_dim
        config['gnn']['edge_feat_dim'] = temp_env.resource_mgr.edge_feat_dim
        config['gnn']['request_feat_dim'] = temp_env.resource_mgr.request_dim
        del temp_env  # 释放资源

        # 2. 初始化智能体
        agent = HRL_DQN_Agent(config)

        # 3. 检查数据文件
        data_file = "expert_data_final.pkl"  # 假设这是 Phase 1 的输出名
        data_path = os.path.join(config['path']['expert_data_dir'], data_file)

        if not os.path.exists(data_path):
            logger.error(f"❌ Expert data not found at {data_path}. Run Phase 1 first.")
            return

        trainer = Phase2ILTrainer(
            agent=agent,
            expert_data_path=data_path,
            output_dir=config['path']['ckpt_dir'],
            config=config.get('il', {})  # 对应 phase2.yaml 中的 il 块
        )
        trainer.run()

    # ------------------------------------
    # Phase 3: 强化学习微调 (RL Fine-tuning)
    # ------------------------------------
    elif args.phase == 'phase3':
        logger.info(">>> Initializing Phase 3: RL Fine-tuning")

        # 1. 初始化环境
        env = SFC_HIRL_Env(config, use_gnn=True)

        # 2. 注入动态维度到配置
        config['gnn']['node_feat_dim'] = env.resource_mgr.node_feat_dim
        config['gnn']['edge_feat_dim'] = env.resource_mgr.edge_feat_dim
        config['gnn']['request_feat_dim'] = env.resource_mgr.request_dim

        # 3. 初始化智能体
        agent = HRL_DQN_Agent(config)

        # 4. 加载 Phase 2 预训练模型 (可选)
        pretrained_path = os.path.join(config['path']['ckpt_dir'], "il_model_final.pth")
        if os.path.exists(pretrained_path):
            logger.info(f"📥 Loading pretrained IL model: {pretrained_path}")
            agent.load(pretrained_path)
        else:
            logger.warning("⚠️ No pretrained model found. Starting RL from scratch (Random init).")

        # 5. 开始训练
        trainer = Phase3RLTrainer(
            env=env,
            agent=agent,
            output_dir=config['path']['ckpt_dir'],
            config=config  # 传入完整 config
        )
        trainer.run()


if __name__ == "__main__":
    main()