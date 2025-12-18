import pickle
import os
import sys


def check_data_integrity(requests_path, events_path):
    print("=" * 60)
    print("🔍 数据完整性与逻辑检查工具")
    print("=" * 60)
    print(f"📂 正在读取文件:")
    print(f"   1. Requests: {requests_path}")
    print(f"   2. Events:   {events_path}")

    # 1. 加载文件
    if not os.path.exists(requests_path) or not os.path.exists(events_path):
        print("\n❌ 错误: 文件不存在，请先运行 main_generate.py 生成数据。")
        return

    try:
        with open(requests_path, 'rb') as f:
            requests = pickle.load(f)
        with open(events_path, 'rb') as f:
            events = pickle.load(f)
    except Exception as e:
        print(f"\n❌ 读取失败: {e}")
        return

    # 2. 检查 Requests ID 连续性
    print("\n[Step 1] 检查 Requests ID 连续性...")
    id_errors = 0
    for idx, req in enumerate(requests):
        expected_id = idx + 1
        if req['id'] != expected_id:
            if id_errors < 5:
                print(f"   ❌ ID 错位! Index={idx}, Expect={expected_id}, Got={req['id']}")
            id_errors += 1

    if id_errors == 0:
        print(f"   ✅ ID 检查通过 (共 {len(requests)} 个请求, ID 1~{len(requests)})")
    else:
        print(f"   🚫 ID 检查失败: 发现 {id_errors} 个 ID 错误")

    # 3. 检查 Events 逻辑 (核心)
    print("\n[Step 2] 检查 Events 调度逻辑...")
    print(f"   - 时间轴总长度: {len(events)} time steps")

    seen_ids = set()
    event_errors = 0
    duplicate_errors = 0

    for t, event in enumerate(events):
        arrive_list = event.get('arrive_event', [])

        # 检查每个到达事件
        for req_id in arrive_list:
            # 🚨 检查 1: ID 是否重复出现 (死循环的根源)
            if req_id in seen_ids:
                if duplicate_errors < 5:
                    print(f"   ❌ [致命错误] ID={req_id} 重复出现! 上次已处理过，现在 t={t} 又出现了。")
                duplicate_errors += 1
                event_errors += 1
            seen_ids.add(req_id)

            # 🚨 检查 2: ID 是否有效
            if req_id < 1 or req_id > len(requests):
                print(f"   ❌ [越界错误] t={t}, 发现了无效 ID={req_id} (有效范围 1~{len(requests)})")
                event_errors += 1
                continue

            # 🚨 检查 3: 时间是否匹配
            # 获取对应的请求对象
            req = requests[req_id - 1]
            req_arrive_step = req.get('arrive_time_step')

            # 允许有 1 个时间步的误差 (浮点数 ceil 导致)
            if req_arrive_step is not None and abs(req_arrive_step - t) > 1:
                print(f"   ⚠️ [时间不一致] Req {req_id}: Request记录是 t={req_arrive_step}, 但在 Events t={t} 触发")
                # 这通常不是致命错误，只要 ID 唯一即可

    # 4. 总结
    print("\n" + "-" * 60)
    print("📊 检查报告")
    print("-" * 60)

    if duplicate_errors > 0:
        print(f"🔴 结果: 严重失败! 发现 {duplicate_errors} 个重复 ID。")
        print("   原因: events.pkl 生成逻辑有误，或者是旧文件没删除。")
        print("   后果: 会导致 Phase 1 死循环处理同一个请求。")
    elif event_errors > 0:
        print(f"🟠 结果: 存在 {event_errors} 个逻辑错误，建议重新生成。")
    else:
        # 检查是否所有请求都安排了
        missing = len(requests) - len(seen_ids)
        if missing == 0:
            print("🟢 结果: 完美! 所有检查通过。可以放心运行仿真。")
        else:
            print(f"🟡 结果: 通过，但有 {missing} 个请求未在 Events 中出现 (可能是时间超过了 events 数组长度)。")
            print("   (这对 Phase 1 影响不大，只会少跑几个请求)")


# --- 执行区 ---
if __name__ == "__main__":
    # 请根据您的实际路径修改这里
    # 如果您的 main_generate.py 输出到 ./data/input_dir
    BASE_DIR = "./data_output"

    # 或者是 ./data_output (取决于您上次运行 main_generate.py 时的设置)
    if not os.path.exists(BASE_DIR):
        BASE_DIR = "./data_output"

    REQ_FILE = os.path.join(BASE_DIR, "phase1_requests.pkl")
    EVT_FILE = os.path.join(BASE_DIR, "phase1_events.pkl")

    check_data_integrity(REQ_FILE, EVT_FILE)