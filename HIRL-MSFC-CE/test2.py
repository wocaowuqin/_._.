"""快速测试 Phase 3 环境"""

import pickle
from pathlib import Path
import hyperparameters as H
from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN

# 创建环境
env = SFC_HIRL_Env_GNN(
    input_dir=H.INPUT_DIR,
    topo=H.TOPOLOGY_MATRIX,
    dc_nodes=H.DC_NODES,
    capacities=H.CAPACITIES,
    use_gnn=True
)

# 加载 Phase 3 数据
with open("phase3_requests.pkl", 'rb') as f:
    env.request_list = pickle.load(f)
with open("phase3_events.pkl", 'rb') as f:
    env.event_list = pickle.load(f)

print(f"✅ 环境数据加载成功：{len(env.request_list)} 个请求")

# 测试 reset_request
print("\n🔍 测试 reset_request:")
req, state = env.reset_request()

if req is None:
    print("❌ reset_request 返回 None!")
    print("   可能原因：request_list 或 event_list 格式不对")
else:
    print(f"✅ 成功获取请求:")
    print(f"   ID: {req.get('id')}")
    print(f"   Source: {req.get('source')}")
    print(f"   Dest: {req.get('dest')}")

    # 测试 unadded_dest_indices
    print("\n🔍 测试 unadded_dest_indices:")
    if hasattr(env, 'unadded_dest_indices'):
        unadded = list(env.unadded_dest_indices)
        print(f"✅ unadded_dest_indices: {unadded}")

        if not unadded:
            print("❌ 列表为空！无法训练")
    else:
        print("❌ 环境没有 unadded_dest_indices 属性")

    # 测试 get_valid_low_level_actions
    print("\n🔍 测试 get_valid_low_level_actions:")
    try:
        valid_actions = env.get_valid_low_level_actions()
        print(f"✅ 有效动作数: {len(valid_actions)}")
        if len(valid_actions) > 0:
            print(f"   前 10 个动作: {valid_actions[:10]}")
        else:
            print("❌ 没有有效动作！")
    except Exception as e:
        print(f"❌ 获取有效动作失败: {e}")