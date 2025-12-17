import pickle
import os
import sys
import numpy as np

# 设置数据路径 (根据您之前的日志)
DATA_DIR = r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir"
REQ_FILE = "phase1_requests.pkl"
EVT_FILE = "phase1_events.pkl"


def test_events():
    print(f"🚀 开始测试事件数据加载...")
    print(f"📂 数据目录: {DATA_DIR}")

    req_path = os.path.join(DATA_DIR, REQ_FILE)
    evt_path = os.path.join(DATA_DIR, EVT_FILE)

    # 1. 检查文件是否存在
    if not os.path.exists(req_path) or not os.path.exists(evt_path):
        print("❌ 错误: 找不到数据文件！请检查路径。")
        return

    # 2. 加载请求数据
    try:
        with open(req_path, 'rb') as f:
            requests = pickle.load(f)
        print(f"✅ 请求文件加载成功: {len(requests)} 条请求")

        # 提取所有请求 ID
        req_ids = {int(r['id']) for r in requests}
        print(f"   请求 ID 样例 (前5个): {sorted(list(req_ids))[:5]}")
        print(f"   请求 ID 范围: {min(req_ids)} ~ {max(req_ids)}")

    except Exception as e:
        print(f"❌ 请求文件读取失败: {e}")
        return

    print("-" * 50)

    # 3. 加载事件数据
    try:
        with open(evt_path, 'rb') as f:
            raw_events = pickle.load(f)
        print(f"✅ 事件文件加载成功: {len(raw_events)} 个时间步")
    except Exception as e:
        print(f"❌ 事件文件读取失败: {e}")
        return

    # 4. 深度分析事件结构
    print("\n🔍 正在分析事件结构...")

    valid_steps = 0
    all_event_ids = set()
    first_valid_step = None

    for i, evt in enumerate(raw_events):
        arrive_ids = []

        # 兼容性解析逻辑 (与 data_loader.py 保持一致)
        if isinstance(evt, dict):
            # 尝试所有可能的键名
            arrive_ids = evt.get('arrive_event', evt.get('arrive', evt.get('arrived', [])))
        elif isinstance(evt, (list, tuple, np.ndarray)):
            if len(evt) >= 1:
                arrive_ids = evt[0]

        # 转换为列表
        if isinstance(arrive_ids, (np.ndarray, list, tuple)):
            arrive_ids = np.array(arrive_ids, dtype=int).flatten().tolist()

        if len(arrive_ids) > 0:
            valid_steps += 1
            all_event_ids.update(arrive_ids)
            if first_valid_step is None:
                first_valid_step = (i, arrive_ids)

    print(f"   非空时间步数量: {valid_steps} / {len(raw_events)}")

    if valid_steps == 0:
        print("❌ 严重错误: 所有时间步的 arrive 列表都是空的！")
        print("   请检查 .pkl 文件的键名是否为 'arrive_event', 'arrive' 或 'arrived'")
        # 打印第0步的原始结构供调试
        print(f"   第0步原始数据: {raw_events[0]}")
        return

    print(f"   检测到的事件 ID 总数: {len(all_event_ids)}")
    print(f"   事件 ID 样例 (前5个): {sorted(list(all_event_ids))[:5]}")
    print(f"   事件 ID 范围: {min(all_event_ids)} ~ {max(all_event_ids)}")

    if first_valid_step:
        step_idx, ids = first_valid_step
        print(f"   第一个有效时间步: Step {step_idx}, 包含 ID: {ids}")

    print("-" * 50)

    # 5. ID 对齐测试
    print("\n⚖️ 正在进行 ID 对齐测试...")

    intersection = req_ids.intersection(all_event_ids)

    if len(intersection) > 0:
        print(f"✅ ID 匹配成功！共有 {len(intersection)} 个请求能被事件触发。")
    else:
        print("❌ ID 不匹配！")
        print("   尝试自动修复逻辑检测...")

        # 模拟 -1 修复
        shifted_down = {x - 1 for x in all_event_ids}
        match_down = len(req_ids.intersection(shifted_down))

        # 模拟 +1 修复
        shifted_up = {x + 1 for x in all_event_ids}
        match_up = len(req_ids.intersection(shifted_up))

        if match_down > 0:
            print(f"   💡 建议: 如果将 Event ID 减 1，可以匹配 {match_down} 个请求。")
            print("   -> DataLoader 中的自动修复逻辑应该能生效。")
        elif match_up > 0:
            print(f"   💡 建议: 如果将 Event ID 加 1，可以匹配 {match_up} 个请求。")
            print("   -> DataLoader 中的自动修复逻辑应该能生效。")
        else:
            print("   ❌ 即使偏移 ±1 也无法匹配。请检查数据源生成逻辑。")


if __name__ == "__main__":
    test_events()