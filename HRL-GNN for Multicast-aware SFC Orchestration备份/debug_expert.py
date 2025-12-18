import sys
import os
import numpy as np
import logging
from pathlib import Path

# 配置简单的日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DebugExpert")


def debug_expert_integration():
    print("=" * 60)
    print("🔍 专家模块集成诊断 (Expert Integration Diagnostic)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. 检测专家模块导入
    # ---------------------------------------------------------
    print("\n[1] 正在尝试导入专家模块 (expert_msfce)...")
    try:
        # 尝试添加当前目录到路径
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        import expert_msfce
        from expert_msfce import MSFCE_Solver

        print(f"   ✅ 导入成功！模块文件位置: {expert_msfce.__file__}")

        # 检查是否是占位符类（通过检查是否有 solve_request_for_expert 方法）
        if not hasattr(MSFCE_Solver, 'solve_request_for_expert'):
            print("   ❌ 警告：导入的类似乎是旧版本，缺少 solve_request_for_expert 方法！")
    except ImportError as e:
        print(f"   ❌ 导入失败 (ImportError): {e}")
        print("   -> 可能是文件名不对，或者缺少依赖包 (如 scipy)")
        return
    except Exception as e:
        print(f"   ❌ 导入时发生未知错误: {e}")
        return

    # ---------------------------------------------------------
    # 2. 初始化专家实例
    # ---------------------------------------------------------
    print("\n[2] 正在初始化专家实例...")

    # 模拟配置
    topo_size = 28
    topo = np.zeros((topo_size, topo_size))  # 假拓扑，只要尺寸对就行
    dc_nodes = [1, 2, 3]
    capacities = {'cpu': 100, 'memory': 100, 'bandwidth': 100}

    # 寻找 .mat 文件
    mat_path = Path("data/input_dir/US_Backbone_path.mat")
    if not mat_path.exists():
        print(f"   ⚠️ 未找到默认路径: {mat_path}")
        # 尝试绝对路径
        mat_path = Path(
            r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\US_Backbone_path.mat")

    if not mat_path.exists():
        print(f"   ❌ 严重错误：找不到 .mat 数据库文件: {mat_path}")
        print("   -> 专家无法初始化")
        return

    try:
        expert = MSFCE_Solver(mat_path, topo, dc_nodes, capacities)
        print("   ✅ 专家初始化成功！")
        print(f"   -> 专家内部节点数: {expert.node_num}")
        print(f"   -> 预计算路径缓存大小: {len(expert._path_cache) if hasattr(expert, '_path_cache') else 'Unknown'}")
    except Exception as e:
        print(f"   ❌ 初始化崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. 模拟一次请求 (测试 ID 对齐)
    # ---------------------------------------------------------
    print("\n[3] 正在模拟请求处理 (ID Alignment Test)...")

    # 构造一个环境视角的请求 (0-based)
    # 假设 Source=0 (对应物理节点1), Dest=[5] (对应物理节点6)
    env_req = {
        'id': 999,
        'source': 0,
        'dest': [5],
        'vnf': [1],
        'bw_origin': 1.0,
        'cpu_origin': [1.0],
        'memory_origin': [1.0]
    }

    # 构造网络状态
    network_state = {
        'cpu': np.full(28, 100.0),  # 资源充足
        'mem': np.full(28, 100.0),
        'bw': np.full(100, 100.0)
    }

    print(f"   原始请求 (Env 0-based): Source={env_req['source']}, Dest={env_req['dest']}")

    # 模拟 PolicyHelper 里的转换逻辑
    expert_req = env_req.copy()
    expert_req['source'] = int(env_req['source']) + 1
    expert_req['dest'] = [int(d) + 1 for d in env_req['dest']]

    print(f"   转换后请求 (Expert 1-based): Source={expert_req['source']}, Dest={expert_req['dest']}")
    print("   -> 正在调用 expert.solve_request_for_expert()...")

    try:
        tree, traj = expert.solve_request_for_expert(expert_req, network_state)

        if tree is not None:
            print("   ✅ 专家返回了方案！(SUCCESS)")
            print(f"   -> Trajectory: {traj}")
            print("   -> 结论：ID转换逻辑有效，专家工作正常。")
        else:
            print("   ❌ 专家返回 None (FAILED)")
            print("   -> 可能原因：资源检查失败、路径不可达、或者 expert_msfce 内部逻辑拒绝了请求。")

            # 尝试不转换直接传（反向测试）
            print("\n   [Test B] 尝试不转换 ID 直接传 (0-based)...")
            tree_b, traj_b = expert.solve_request_for_expert(env_req, network_state)
            if tree_b:
                print("   ⚠️ 竟然成功了？说明专家内部可能已经是 0-based，或者我们不需要 +1。")
            else:
                print("   ❌ 依然失败。问题可能出在资源检查或路径库。")

    except Exception as e:
        print(f"   ❌ 调用过程崩溃: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_expert_integration()