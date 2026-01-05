# file: test_phase2.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 模型测试脚本
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluator import Phase2Evaluator
from main import setup_environment, setup_agent, load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase2_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主测试函数"""
    logger.info("=" * 70)
    logger.info("🧪 Phase 2 模型评估")
    logger.info("=" * 70)

    # 1. 加载配置
    config = load_config()

    # 2. 设置环境
    logger.info("🔧 初始化环境...")
    env = setup_environment(config)

    # 3. 设置Agent
    logger.info("🤖 初始化Agent...")
    agent = setup_agent(config, env)

    # 4. 创建评估器
    evaluator = Phase2Evaluator(agent, env, config)

    # 5. 加载最佳模型
    checkpoint_path = "outputs/checkpoints/il_model_best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "outputs/checkpoints/il_model_final.pth"

    if os.path.exists(checkpoint_path):
        epoch, val_loss = evaluator.load_model(checkpoint_path)
        logger.info(f"📈 模型信息: Epoch {epoch}, Val Loss: {val_loss:.6f}")
    else:
        logger.error(f"❌ 模型文件不存在: {checkpoint_path}")
        return

    # 6. 在数据集上评估
    expert_data_path = "outputs/expert/expert_data_final.pkl"
    if os.path.exists(expert_data_path):
        logger.info("\n" + "=" * 60)
        logger.info("📊 数据集评估")
        logger.info("=" * 60)

        dataset_results = evaluator.evaluate_on_dataset(expert_data_path)

        # 保存数据集评估结果
        import json
        with open("outputs/evaluation/dataset_results.json", "w") as f:
            json.dump(dataset_results, f, indent=2, default=str)
    else:
        logger.warning(f"⚠️  专家数据不存在: {expert_data_path}")

    # 7. 在环境中评估
    logger.info("\n" + "=" * 60)
    logger.info("🎮 环境评估")
    logger.info("=" * 60)

    env_results = evaluator.evaluate_in_environment(num_episodes=20)

    # 保存环境评估结果
    import json
    with open("outputs/evaluation/environment_results.json", "w") as f:
        json.dump(env_results, f, indent=2, default=str)

    # 8. 可视化结果
    logger.info("\n" + "=" * 60)
    logger.info("📈 生成可视化")
    logger.info("=" * 60)

    evaluator.visualize_results("outputs/evaluation/visualizations")

    # 9. 生成评估报告
    generate_evaluation_report(dataset_results, env_results)

    logger.info("=" * 70)
    logger.info("✅ Phase 2 评估完成")
    logger.info("=" * 70)


def generate_evaluation_report(dataset_results, env_results):
    """生成评估报告"""
    report = f"""
{'=' * 70}
Phase 2 模型评估报告
{'=' * 70}

📊 数据集评估结果:
    • 总体准确率: {dataset_results.get('accuracy', 0):.4f}
    • 精确率 (Precision): {dataset_results.get('precision', 0):.4f}
    • 召回率 (Recall): {dataset_results.get('recall', 0):.4f}
    • F1分数: {dataset_results.get('f1_score', 0):.4f}

🎮 环境评估结果:
    • 成功率: {env_results.get('success_rate', 0):.4f}
    • 平均奖励: {env_results.get('avg_reward', 0):.4f}
    • 平均步数: {env_results.get('avg_steps', 0):.2f}
    • 完成率: {env_results.get('completion_rate', 0):.4f}
    • 阻塞率: {env_results.get('blocking_rate', 0):.4f}

📈 评估标准:
    • 优秀 (Excellent): 准确率 > 0.85
    • 良好 (Good): 准确率 0.70 - 0.85
    • 一般 (Fair): 准确率 0.50 - 0.70
    • 需要改进 (Needs Improvement): 准确率 < 0.50

💡 改进建议:
    1. 如果准确率 < 0.50: 考虑重新训练或调整模型架构
    2. 如果准确率 0.50-0.70: 尝试增加训练数据或数据增强
    3. 如果准确率 0.70-0.85: 模型表现良好，可尝试微调
    4. 如果准确率 > 0.85: 模型表现优秀，可直接使用

{'=' * 70}
"""

    print(report)

    # 保存报告
    report_path = Path("outputs/evaluation/evaluation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"📄 评估报告已保存: {report_path}")


def quick_test_model():
    """快速测试模型"""
    import torch

    # 创建一个简单的测试
    config = load_config()
    env = setup_environment(config)
    agent = setup_agent(config, env)

    # 加载模型
    checkpoint_path = "outputs/checkpoints/il_model_final.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])

    agent.policy_net.eval()

    # 测试几个推理
    logger.info("🧪 快速推理测试...")

    # 创建一个随机输入
    num_nodes = config.get('num_nodes', 28)
    node_feat_dim = config.get('node_feat_dim', 17)

    # 创建模拟数据
    x = torch.randn(num_nodes, node_feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(num_nodes * 2, 5)
    req_vec = torch.randn(1, 24)

    # 运行推理
    with torch.no_grad():
        output = agent.policy_net(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            req_vec=req_vec,
            batch=torch.zeros(num_nodes, dtype=torch.long)
        )

    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.softmax(logits, dim=1)

    logger.info(f"📊 推理测试完成:")
    logger.info(f"  - Logits shape: {logits.shape}")
    logger.info(f"  - Probs shape: {probs.shape}")
    logger.info(f"  - Max prob: {probs.max().item():.4f}")
    logger.info(f"  - Min prob: {probs.min().item():.4f}")

    # 预测动作
    predicted_action = torch.argmax(logits, dim=1)
    logger.info(f"  - Predicted actions: {predicted_action.tolist()[:10]}...")

    return agent.policy_net


if __name__ == "__main__":
    main()