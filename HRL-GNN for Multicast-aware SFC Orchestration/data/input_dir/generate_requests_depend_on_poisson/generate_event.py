import pickle
import os


def process_single_file(input_name, output_name):
    """处理单个请求文件生成事件列表"""
    input_path = os.path.join('./data_output', input_name)
    output_path = os.path.join('./data_output', output_name)

    if not os.path.exists(input_path):
        print(f"⚠️ 跳过: 未找到 {input_path}")
        return

    print(f"正在处理: {input_name} -> {output_name} ...")

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    # 找出最大时间步
    max_time_step = 0
    for req in requests_list:
        if req['leave_time_step'] > max_time_step:
            max_time_step = req['leave_time_step']

    # 初始化事件列表
    event_list = []
    for t in range(max_time_step + 2):
        event_list.append({
            'time_step': t,
            'arrive_event': [],
            'leave_event': []
        })

    # 填充事件
    for req in requests_list:
        t_arr = req['arrive_time_step']
        t_leave = req['leave_time_step']

        if t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req['id'])

        if t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req['id'])

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f"✓ 已生成: {output_path} (覆盖 {len(requests_list)} 条请求)")


def generate_events():
    print("=" * 60)
    print("生成事件列表 (Event Generation)")
    print("=" * 60)

    # 1. 处理 Phase 1 (专家/监督学习用)
    process_single_file('phase1_requests.pkl', 'phase1_events.pkl')

    # 2. 处理 Phase 3 (RL微调用)
    process_single_file('phase3_requests.pkl', 'phase3_events.pkl')

    # 3. 处理默认文件 (兼容旧代码)
    process_single_file('sorted_requests.pkl', 'event_list.pkl')

    print("\n所有事件列表生成完毕！")


if __name__ == '__main__':
    generate_events()