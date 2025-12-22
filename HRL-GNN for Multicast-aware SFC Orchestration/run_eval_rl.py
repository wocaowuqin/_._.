import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import time
import inspect
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent.agent import Agent
from utils.config_utils import load_config

# ================= 你的配置区域 =================
TEST_DATA_PATH = 'data/input_dir/generate_requests_depend_on_poisson/data_output/test_requests.pkl'
MODEL_PATH = 'outputs/checkpoints/rl_model_final.pth'
CONFIG_NAME = 'phase3'


# ===============================================

def probe_model_dimensions(model_path):
    """
    预读取模型文件，精准侦测训练时使用的特征维度
    """
    print(f"🔍 [维度侦测] 正在扫描模型文件: {model_path}")
    dims = {}
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'q_network' in checkpoint:
                state_dict = checkpoint['q_network']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 遍历权重寻找特征层
        for key, weight in state_dict.items():
            # 1. 检测 Node Features
            if 'node_lin.weight' in key:
                # 只有当维度较小时才认为是输入层 (防止误判 hidden_dim)
                if weight.shape[1] < 100:
                    dims['node_feat'] = weight.shape[1]
                    print(f"   👉 [锁定] Node Dim: {dims['node_feat']} (from {key})")

            # 2. 检测 Edge Features
            if 'lin_edge.weight' in key:
                if weight.shape[1] < 100:
                    dims['edge_feat'] = weight.shape[1]
                    print(f"   👉 [锁定] Edge Dim: {dims['edge_feat']} (from {key})")

            # 3. 检测 Request Features (🔥 核心修复)
            # 只认准 .0.weight (第一层)，防止被后面的层覆盖
            if 'req_modulator' in key and 'weight' in key:
                # 如果包含 .0. 或者不包含 .数字. (单层)，且维度合理
                if '.0.weight' in key or not any(char.isdigit() for char in key.split('.')):
                    if weight.shape[1] < 200:  # 排除 hidden_dim (通常 128/256)
                        dims['req_feat'] = weight.shape[1]
                        print(f"   👉 [锁定] Req Dim:  {dims['req_feat']} (from {key})")

    except Exception as e:
        print(f"⚠️ 维度侦测失败: {e}")

    return dims


def run_rl_evaluation():
    print("=" * 60)
    print("🤖 启动 RL 模型评估 (Inference Mode)")
    print(f"📄 测试集: {TEST_DATA_PATH}")
    print(f"💾 模型: {MODEL_PATH}")
    print("=" * 60)

    # 1. 加载配置
    try:
        config = load_config(CONFIG_NAME)
        if 'eval' not in config: config['eval'] = {}
        config['eval']['device'] = 'cpu'
        config['device'] = 'cpu'
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return

    # 2. 初始化环境
    try:
        env = SFC_HIRL_Env(config)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    # 3. 加载测试数据
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误: 找不到测试集文件 {TEST_DATA_PATH}")
        return

    try:
        with open(TEST_DATA_PATH, 'rb') as f:
            test_requests = pickle.load(f)
        env.data_loader.requests = test_requests
        env.data_loader.reset()
        print(f"✅ 成功加载 {len(test_requests)} 条测试请求")
    except Exception as e:
        print(f"❌ 读取测试集失败: {e}")
        return

    # 4. 🔥 核心修复：根据模型文件覆盖 Config 维度
    if os.path.exists(MODEL_PATH):
        detected_dims = probe_model_dimensions(MODEL_PATH)

        if 'gnn' not in config: config['gnn'] = {}

        # 强制覆盖所有可能的配置键，确保万无一失
        if 'node_feat' in detected_dims:
            val = detected_dims['node_feat']
            config['gnn']['node_feat'] = val
            config['gnn']['node_feat_dim'] = val
            config['gnn']['node_input_dim'] = val

        if 'edge_feat' in detected_dims:
            val = detected_dims['edge_feat']
            config['gnn']['edge_feat'] = val
            config['gnn']['edge_feat_dim'] = val
            config['gnn']['edge_input_dim'] = val

        if 'req_feat' in detected_dims:
            val = detected_dims['req_feat']
            config['gnn']['req'] = val
            config['gnn']['req_dim'] = val
            config['gnn']['req_feat_dim'] = val
            config['gnn']['request_dim'] = val  # 穷举所有可能的命名

    # 注入环境维度
    state_dim = env.resource_mgr.STATE_VECTOR_SIZE
    action_dim = env.action_space.n

    if 'model' not in config: config['model'] = {}
    config['model']['state_dim'] = state_dim
    config['model']['action_dim'] = action_dim
    config['state_dim'] = state_dim
    config['action_dim'] = action_dim

    print(f"ℹ️ 初始化 HRL_DQN_Agent (Phase 3)...")
    try:
        # 指定 phase=3 以使用 q_network
        agent = Agent(config, phase=3, low_action_dim=action_dim, high_action_dim=10)
        print("✅ Agent 初始化成功")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        return

    # 5. 加载模型权重
    device = torch.device('cpu')
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
            state_dict = checkpoint['q_network']
            print("👉 加载 'q_network' 权重...")
        else:
            state_dict = checkpoint
            print("👉 直接加载 checkpoint 权重...")

        if hasattr(agent, 'q_network'):
            agent.q_network.load_state_dict(state_dict)
            agent.q_network.to(device)
            agent.q_network.eval()
            print(f"✅ 成功加载权重到 agent.q_network")
        else:
            print("❌ Agent 没有 q_network 属性。")
            return

    except RuntimeError as e:
        print(f"❌ 权重加载失败: {e}")
        if 'req_modulator' in str(e):
            print("\n💡 诊断: 依然是 Request 维度不匹配。")
            print("   请确认 'detected_dims' 是否正确打印出了 24。")
        return
    except Exception as e:
        print(f"❌ 加载出错: {e}")
        return

    # 6. 开始推演
    total_episodes = len(test_requests)
    success_count = 0
    start_time = time.time()

    state, info = env.reset(options={'phase': 'test'})
    print(f"\n🚀 开始处理 {total_episodes} 个请求...")
    print_interval = max(1, total_episodes // 10)

    for i in range(total_episodes):
        done = False
        while not done:
            try:
                high_mask = env.get_high_level_action_mask()
                low_mask = env.get_low_level_action_mask()
                high_act, low_act = agent.select_action(state, masks=(high_mask, low_mask), epsilon=0.0)

                next_state, reward, done, truncated, info = env.step(low_act)
                state = next_state

                if done and info.get('success'):
                    success_count += 1
            except RuntimeError as re:
                if "mat1 and mat2 shapes cannot be multiplied" in str(re):
                    print(f"\n❌ [运行时数据不匹配] 虽然模型加载成功，但数据生成器产生的维度不对！")
                    print(f"   模型期望 Req Dim: 24")
                    print(f"   当前代码生成 Req Dim: 6 (默认值)")
                    print(f"💡 必须修改 GNNFeatureBuilder 代码，使其生成 24 维特征，或者使用旧版代码。")
                    return
                raise re

        state, info = env.reset()
        if (i + 1) % print_interval == 0:
            print(f"进度: {i + 1}/{total_episodes} | 当前接受率: {success_count / (i + 1):.2%}")

    duration = time.time() - start_time
    avg_time = (duration / total_episodes) * 1000

    print("\n" + "=" * 60)
    print("📊 RL 模型最终评估报告")
    print("=" * 60)
    print(f"测试请求数: {total_episodes}")
    print(f"成功接受数: {success_count}")
    print(f"🏆 最终接受率: {success_count / total_episodes:.2%}")
    print(f"⏱️ 平均耗时:   {avg_time:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    run_rl_evaluation()