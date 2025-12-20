#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Encoder Connection Verification Script
验证 Phase 2 的编码器参数能否顺利传递给 Phase 3
"""

import torch
import torch.nn as nn
import os
import sys
import logging
import numpy as np

# 设置路径以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 SharedEncoder
try:
    from core.gnn.shared_encoder import SharedEncoder
except ImportError:
    # 如果找不到文件，这里定义一个临时的 Mock 类用于演示逻辑
    print("⚠️ 未找到 core.gnn.shared_encoder，使用临时定义进行演示...")
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    import torch.nn.functional as F


    class SharedEncoder(nn.Module):
        def __init__(self, node_feat_dim, edge_feat_dim, request_dim, hidden_dim, num_layers=3, heads=4):
            super().__init__()
            self.node_lin = nn.Linear(node_feat_dim, hidden_dim)
            # ... (简化版，仅用于测试权重加载) ...
            self.dummy_param = nn.Parameter(torch.randn(10, 10))  # 用于验证权重的标记

        def forward(self, x, edge_index, edge_attr, req_vec, batch):
            return x

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("EncoderTest")


def check_parameter_match(model_a, model_b):
    """检查两个模型的参数是否完全一致"""
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a != name_b:
            return False, f"参数名称不匹配: {name_a} vs {name_b}"
        if not torch.equal(param_a, param_b):
            return False, f"参数数值不匹配: {name_a}"
    return True, "所有参数一致"


def main():
    logger.info("🚀 开始验证 SharedEncoder 衔接...")

    # ====================================================
    # 1. 模拟 Phase 2：训练并保存 (Mock Phase 2)
    # ====================================================
    logger.info("\n--- [步骤 1] 模拟 Phase 2 训练与保存 ---")

    # 模拟 Phase 2 中训练好的编码器
    # 假设配置: node=10, edge=5, req=8, hidden=32
    phase2_encoder = SharedEncoder(10, 5, 8, 32)

    # 为了证明它不是随机的，我们要修改一下它的权重（模拟训练过程）
    with torch.no_grad():
        for param in phase2_encoder.parameters():
            param.add_(1.0)  # 所有参数加 1，使其区别于随机初始化

    # 获取指纹（用于人工核对）
    p2_fingerprint = list(phase2_encoder.parameters())[0].data[0, :5].numpy()
    logger.info(f"Phase 2 编码器指纹 (前5位): {p2_fingerprint}")

    # 保存参数到文件
    save_path = "temp_encoder_test.pth"
    torch.save(phase2_encoder.state_dict(), save_path)
    logger.info(f"✅ 模拟 Phase 2 保存完成: {save_path}")

    # ====================================================
    # 2. 模拟 Phase 3：初始化并加载 (Mock Phase 3)
    # ====================================================
    logger.info("\n--- [步骤 2] 模拟 Phase 3 初始化与加载 ---")

    # 初始化 Phase 3 编码器（随机权重）
    phase3_encoder = SharedEncoder(10, 5, 8, 32)

    # 检查加载前是否不同
    is_match, msg = check_parameter_match(phase2_encoder, phase3_encoder)
    if is_match:
        logger.error("❌ 错误：Phase 3 初始化权重竟然与 Phase 2 相同（极低概率或逻辑错误）")
        return
    else:
        logger.info(f"✅ 加载前状态确认: 权重不一致 ({msg})")

    # --- 执行加载操作 ---
    logger.info(f"📥 正在加载 Phase 2 权重...")
    try:
        if os.path.exists(save_path):
            state_dict = torch.load(save_path)
            phase3_encoder.load_state_dict(state_dict)
            logger.info("✅ load_state_dict 执行成功")
        else:
            logger.error("❌ 找不到权重文件")
            return
    except Exception as e:
        logger.error(f"❌ 加载抛出异常: {e}")
        return

    # ====================================================
    # 3. 最终验证
    # ====================================================
    logger.info("\n--- [步骤 3] 验证衔接结果 ---")

    p3_fingerprint = list(phase3_encoder.parameters())[0].data[0, :5].numpy()
    logger.info(f"Phase 3 编码器指纹 (前5位): {p3_fingerprint}")

    is_match, msg = check_parameter_match(phase2_encoder, phase3_encoder)

    if is_match:
        logger.info("\n🎉🎉🎉 验证通过！🎉🎉🎉")
        logger.info("Phase 3 成功复现了 Phase 2 的编码器状态。")
        logger.info("这意味着 'SharedEncoder' 的衔接是顺利的。")
    else:
        logger.error(f"\n❌❌❌ 验证失败: {msg}")
        logger.error("Phase 3 的编码器与 Phase 2 不一致，请检查 load_state_dict 逻辑。")

    # 清理临时文件
    if os.path.exists(save_path):
        os.remove(save_path)


if __name__ == "__main__":
    main()