#!/usr/bin/env python3
"""监控 Phase 2 训练过程 (修复维度匹配版)"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pathlib import Path

# 增加项目路径搜索
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🔍 Phase 2 训练过程监控 (Fixed)")
print("=" * 70)

# ============================================
# 1. 加载配置和模型
# ============================================
print("\n📊 Step 1: 加载模型")

from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env
from core.hrl.agent.agent import HRL_DQN_Agent

# 1. 加载基础配置
config = load_config('phase2')

# 2. 初始化环境
env = SFC_HIRL_Env(config, use_gnn=True)

# 🔥 核心修复：将环境的真实维度注入配置 🔥
print("🔧 正在同步环境维度到配置...")
if 'gnn' not in config: config['gnn'] = {}

# 获取环境中的特征维度
node_dim = env.resource_mgr.node_feat_dim
edge_dim = env.resource_mgr.edge_feat_dim
req_dim = env.resource_mgr.request_dim

# 覆盖配置
config['gnn']['node_feat_dim'] = node_dim
config['gnn']['edge_feat_dim'] = edge_dim
config['gnn']['request_feat_dim'] = req_dim
# 兼容旧键名
config['gnn']['node_feat'] = node_dim
config['gnn']['edge_feat'] = edge_dim
config['gnn']['req'] = req_dim

print(f"   ✅ Node Dim: {node_dim}")
print(f"   ✅ Edge Dim: {edge_dim}")
print(f"   ✅ Req  Dim: {req_dim}")

# 3. 初始化 Agent (现在它会使用正确的 17 维输入层)
agent = HRL_DQN_Agent(
    config,
    high_action_dim=env.NB_HIGH_LEVEL_GOALS,
    low_action_dim=env.NB_LOW_LEVEL_ACTIONS,
    phase=2
)

# 强制模型进入评估模式
agent.policy_net.eval()
print(f"✅ 模型加载成功 (输入层已修正为 {node_dim})")

# ============================================
# 2. 加载数据
# ============================================
print("\n📊 Step 2: 加载数据")

data_path = Path("outputs/expert/expert_data_final.pkl")
if not data_path.exists():
    # 尝试备用路径
    data_path = Path("outputs/expert/expert_transitions_fixed.pkl")

if not data_path.exists():
    print(f"❌ 数据文件不存在: {data_path}")
    exit(1)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 兼容不同的保存格式
if isinstance(data, list):
    transitions = data
    print(f"  识别为 List 格式, 数量: {len(transitions)}")
elif isinstance(data, dict):
    transitions = data.get('success', [])
    print(f"  识别为 Dict 格式, 'success' 数量: {len(transitions)}")
else:
    transitions = []

# 转换为单步数据
samples = []
for trans in transitions:
    # 情况 A: 已经是处理好的 Transition (fix_data.py 生成的)
    if 'state' in trans and isinstance(trans.get('action'), (int, np.integer)):
        samples.append(trans)
        continue

    # 情况 B: 原始 Phase 1 路径数据
    action_data = trans.get('action')
    state = trans.get('state')

    if not isinstance(action_data, dict):
        continue

    path = action_data.get('path', [])
    if len(path) <= 1:
        continue

    # 提取每一步
    for i in range(1, len(path)):
        node = int(path[i]) if isinstance(path[i], np.integer) else path[i]

        # 简单校验
        if node < 0 or node >= 28:
            continue

        samples.append({
            'state': state,
            'action': node,
            'reward': 10.0
        })

print(f"  可用单步样本数: {len(samples)}")

if len(samples) == 0:
    print("❌ 没有有效样本，请检查数据文件！")
    exit(1)

# ============================================
# 3. 测试单个样本的训练
# ============================================
print("\n📊 Step 3: 测试单个样本")

sample = samples[0]
state = sample['state']
action = sample['action']

print(f"  State 类型: {type(state)}")
print(f"  Action: {action}")
if hasattr(state, 'x'):
    print(f"  State.x Shape: {state.x.shape}")

# 前向传播
try:
    state_tensor = state.to(agent.device)

    # 确保 batch 属性存在
    if not hasattr(state_tensor, 'batch') or state_tensor.batch is None:
        state_tensor.batch = torch.zeros(state_tensor.x.size(0), dtype=torch.long, device=agent.device)

    with torch.no_grad():
        logits = agent.policy_net(
            x=state_tensor.x,
            edge_index=state_tensor.edge_index,
            edge_attr=state_tensor.edge_attr if hasattr(state_tensor, 'edge_attr') else None,
            req_vec=state_tensor.req_vec if hasattr(state_tensor, 'req_vec') else torch.zeros(24, device=agent.device),
            batch=state_tensor.batch
        )

    print(f"\n  ✅ 前向传播成功")
    print(f"    Logits 形状: {logits.shape}")

    # 检查预测
    pred_action = logits.argmax(dim=-1).item()
    print(f"    预测 Action: {pred_action}")
    print(f"    真实 Action: {action}")

    # 计算损失
    target = torch.tensor([action], dtype=torch.long, device=agent.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    print(f"    单样本损失: {loss.item():.4f}")

except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    # 打印更详细的维度信息
    print(f"    Model input dim: {agent.policy_net.gnn.encoder.node_lin.weight.shape[1]}")
    print(f"    Data input dim: {state.x.shape[1]}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("诊断完成。如果看到“前向传播成功”，说明维度修复已生效。")
print("=" * 70)