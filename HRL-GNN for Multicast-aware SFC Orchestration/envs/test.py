import numpy as np
import logging
from pathlib import Path
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_pipeline():
    print("=" * 80)
    print("🚀 开始全流程集成测试")
    print("=" * 80)

    # 1. 加载配置 (假设您已经有了 configs/phase3.yaml)
    # 如果没有，先用个假的字典代替
    config = {
        'path': {'input_dir': 'data/expert', 'failure_output_dir': 'data/failure'},
        'topology': {'matrix': np.eye(14), 'dc_nodes': [1, 4]},
        'capacities': {'bandwidth': 100, 'cpu': 100, 'memory': 100},
        'env': {'max_cached_paths': 10, 'nb_high_level_goals': 10, 'nb_low_level_actions': 50}
    }

    # 2. 初始化环境 (开启 GNN 模式)
    print("\n[1] 初始化环境 (GNN Mode)...")
    env = SFC_HIRL_Env(config, use_gnn=True)
    print("✅ 环境初始化成功")

    # 3. 重置环境 (需要有数据文件，否则会报错)
    # 这里我们 mock 一下 load_dataset 以便测试
    env.load_dataset = lambda phase: True
    env.data_loader.get_current_arrivals = lambda: [
        {'id': 1, 'source': 1, 'dest': [2, 3], 'bw_origin': 1.0, 'vnf': [1]}]

    print("\n[2] 重置环境...")
    try:
        # 第一次获取状态
        state = env.reset()
        x, edge_index, edge_attr, req_vec = state  # GNN 状态解包
        print(f"✅ Reset 成功! 节点特征维度: {x.shape}")
    except Exception as e:
        print(f"❌ Reset 失败: {e}")
        return

    # 4. 执行高层动作 (选择目标)
    print("\n[3] 执行 High-Level 动作 (选择目标 0)...")
    state, reward, done, info = env.step_high_level(goal_idx=0)
    print(f"✅ High-Level Step 完成: info={info}")

    # 5. 执行低层动作 (部署路径)
    print("\n[4] 执行 Low-Level 动作 (尝试 Expert/Backup)...")
    # 动作 0 -> k=0, i=0
    state, reward, sub_done, req_done, info = env.step_low_level(action=0)

    print(f"✅ Low-Level Step 完成!")
    print(f"   Reward: {reward}")
    print(f"   Sub Done: {sub_done}")
    print(f"   Info: {info}")

    # 6. 打印统计
    print("\n[5] 打印环境统计...")
    env.print_env_summary()

    print("\n" + "=" * 80)
    print("🎉 集成测试全部通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_full_pipeline()