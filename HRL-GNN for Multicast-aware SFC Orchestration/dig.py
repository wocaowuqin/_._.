import pickle
import numpy as np
import os

FILE_PATH = "outputs/expert/expert_data_final.pkl"


def check_data():
    print("=" * 60)
    print("🕵️‍♂️ 专家数据范围侦测")
    print("=" * 60)

    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在: {FILE_PATH}")
        return

    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)

    # 兼容格式
    if isinstance(data, dict):
        samples = data.get('success', data.get('data', []))
    else:
        samples = data

    min_node = 999
    max_node = -999
    all_nodes = []

    for s in samples:
        action = s.get('action')
        if isinstance(action, dict):
            path = action.get('path', [])
            # 展平 path
            for n in path:
                val = int(n)
                all_nodes.append(val)
                if val < min_node: min_node = val
                if val > max_node: max_node = val
        elif isinstance(action, (int, np.integer)):
            val = int(action)
            all_nodes.append(val)
            if val < min_node: min_node = val
            if val > max_node: max_node = val

    print(f"📊 统计结果 (共 {len(all_nodes)} 个节点记录):")
    print(f"   最小值 (Min Node): {min_node}")
    print(f"   最大值 (Max Node): {max_node}")

    print("-" * 30)
    if min_node == 0:
        print("🚨 结论：数据是 0-based (从 0 开始)！")
        print("❌ 您的代码里 `node - 1` 是错误的，必须删除！")
    elif min_node == 1:
        print("✅ 结论：数据是 1-based (从 1 开始)。")
        print("ok 您的代码里 `node - 1` 是正确的，保留即可。")
    else:
        print(f"⚠️  奇怪的范围，请检查数据。")


if __name__ == "__main__":
    check_data()