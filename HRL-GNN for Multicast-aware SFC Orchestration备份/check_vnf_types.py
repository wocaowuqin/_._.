import pickle
import os
import numpy as np


def check_vnf_types():
    # 常见的数据路径，如果您的路径不同请修改这里
    possible_paths = [
        "data/input_dir/phase1_requests.pkl",
        "data/expert/phase1_requests.pkl",
        r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\phase1_requests.pkl"
    ]

    req_path = None
    for p in possible_paths:
        if os.path.exists(p):
            req_path = p
            break

    if req_path is None:
        print("❌ 找不到 phase1_requests.pkl 文件，请手动修改脚本里的路径。")
        return

    print(f"📂 正在读取: {req_path}")

    try:
        with open(req_path, 'rb') as f:
            requests = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print(f"✅ 成功加载 {len(requests)} 条请求")

    all_vnfs = set()
    max_vnf_len = 0

    # 扫描所有请求
    for req in requests:
        vnfs = req.get('vnf', [])
        if vnfs:
            # 记录链的长度
            max_vnf_len = max(max_vnf_len, len(vnfs))
            # 记录出现过的 VNF ID
            for v in vnfs:
                all_vnfs.add(int(v))

    sorted_vnfs = sorted(list(all_vnfs))

    print("=" * 40)
    print("📊 VNF 数据统计结果")
    print("=" * 40)

    if len(sorted_vnfs) > 0:
        min_id = sorted_vnfs[0]
        max_id = sorted_vnfs[-1]

        print(f"🔹 VNF ID 集合: {sorted_vnfs}")
        print(f"🔹 最小 ID: {min_id}")
        print(f"🔹 最大 ID: {max_id}")
        print(f"🔹 总共有 {len(sorted_vnfs)} 种不同的 VNF")
        print(f"🔹 最长的 SFC 链包含 {max_vnf_len} 个 VNF")

        print("-" * 40)

        # 核心判断
        if max_id >= 8:
            print("🚨 结论: 您的数据中包含 ID >= 8 的 VNF。")
            print("💡 建议: 请在 resource.py 中将 self.K_vnf 设置为至少 8 (或者更大，比如 10)。")
        elif len(sorted_vnfs) > 5 or max_id > 5:
            print(f"🚨 结论: 您的数据中有 {len(sorted_vnfs)} 种 VNF，且最大ID为 {max_id}。")
            print("💡 建议: 请在 resource.py 中将 self.K_vnf 调大 (目前代码里是 5，这显然不够)。")
        else:
            print("✅ 结论: VNF 种类数量看起来在 5 以内，如果报错可能是其他原因。")

    else:
        print("⚠️ 警告: 没有在请求中找到任何 VNF 信息。")


if __name__ == "__main__":
    check_vnf_types()