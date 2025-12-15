import pickle
import numpy as np
from pathlib import Path

# 设置您的数据路径
data_dir = Path(r'E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE\generate_requests_depend_on_poisson\data_output')


def check_phase3():
    print(">>> 正在检查 Phase 3 数据匹配情况...")

    # 1. 加载 Requests
    req_path = data_dir / "phase3_requests.pkl"
    with open(req_path, 'rb') as f:
        reqs = pickle.load(f)
    req_ids = set(r['id'] for r in reqs)
    print(f"Requests 文件: {req_path.name}")
    print(f"  - 请求数量: {len(reqs)}")
    print(f"  - ID 范围: {min(req_ids)} ~ {max(req_ids)}")

    # 2. 加载 Events
    evt_path = data_dir / "phase3_events.pkl"
    with open(evt_path, 'rb') as f:
        evts = pickle.load(f)
    print(f"Events 文件: {evt_path.name}")
    print(f"  - 时间步长度: {len(evts)}")

    # 3. 检查 Events 中的 ID 是否在 Requests 中
    event_ids = set()
    for t, e in enumerate(evts):
        # 兼容两种格式
        ids = e.get('arrive_event', e.get('arrive', []))
        if len(ids) > 0:
            for i in ids:
                event_ids.add(int(i))

    print(f"  - Events 中包含的有效 ID 数量: {len(event_ids)}")

    # 4. 核心检查：交集
    common = req_ids.intersection(event_ids)
    missing = event_ids - req_ids

    print("-" * 30)
    print(f"匹配结果: {len(common)} 个 ID 匹配")
    print(f"不匹配数: {len(missing)} 个 (在 Events 中存在但在 Requests 中找不到)")

    if len(missing) > 0:
        print("❌ 结论: 数据严重不匹配！环境无法找到请求。")
        print(f"    示例缺失 ID: {list(missing)[:5]}")
    elif len(common) == 0:
        print("❌ 结论: 完全不匹配！没有一个请求能对应上。")
    else:
        print("✅ 结论: 数据ID匹配正常。")


if __name__ == '__main__':
    check_phase3()