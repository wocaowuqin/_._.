import numpy as np
import random
import math




# --- 修改1: 确保 VNF 资源系数范围正确 ---
# 为了让 CPU需求在 [1, 24]，当带宽在 [4, 8] 时，系数需为 [0.25, 3.0]
# 为了让 Mem需求在 [1, 16]，当带宽在 [4, 8] 时，系数需为 [0.25, 2.0]
def generate_vnfs_catalog(vnf_type_num=8):
    all_vnf = []
    for vnf_type in range(1, vnf_type_num + 1):
        # 保持之前的逻辑即可，它正好符合要求：
        # 4 * 0.25 = 1 (min), 8 * 3.0 = 24 (max)
        cpu_need = random.random() * 2.75 + 0.25

        # 4 * 0.25 = 1 (min), 8 * 2.0 = 16 (max)
        memory_need = random.random() * 1.75 + 0.25

        vnf = {'type': vnf_type, 'cpu_need': cpu_need, 'memory_need': memory_need}
        all_vnf.append(vnf)
    return all_vnf

def generate_poisson_arrive_time_list(T, lamda):
    """
    生成服从泊松过程的到达时间列表
    :param T: 总时间步
    :param lamda: 到达率
    """
    time_state = 0
    arrive_time_list = []

    while time_state < T:
        # MATLAB: exprnd(1/Lamda) -> Python: random.expovariate(Lamda)
        # expovariate 参数是 lambda (1/mean)，exprnd 参数是 mean (1/lambda)
        interval = random.expovariate(lamda)
        t = time_state + interval
        if t < T:
            time_state = t
            arrive_time_list.append(t)
        else:
            break
    return arrive_time_list


def generate_single_request(req_id, source, dest, all_vnf, max_bw, min_bw, arrive_time, mean_lifetime):
    """
    生成单个业务请求
    """
    vnf_type_num = len(all_vnf)
    vnf_num = 3  # 固定选择3个VNF

    # 随机选择3个不同的VNF类型 (索引从1开始对应type)
    vnf_indices = random.sample(range(vnf_type_num), vnf_num)
    # 获取实际的VNF对象
    selected_vnfs = [all_vnf[i] for i in vnf_indices]
    # 记录VNF类型ID列表
    vnf_types = [v['type'] for v in selected_vnfs]

    # 初始带宽
    bw_origin = random.randint(min_bw, max_bw)

    cpu_origin = []
    memory_origin = []

    for v in selected_vnfs:
        cpu = round(bw_origin * v['cpu_need'])
        mem = round(bw_origin * v['memory_need'])
        cpu_origin.append(cpu)
        memory_origin.append(mem)

    # 请求持续时间 (1 + exp分布), 限制在 [1, 7] 之间 (原代码 >6 重随，即上限为 1+6=7?)
    # 原代码: lifetime > 6 则重随。 exprnd(mean-1).
    while True:
        lifetime = 1 + random.expovariate(1.0 / (mean_lifetime - 1))
        if lifetime <= 6:
            break

    leave_time = arrive_time + lifetime

    # 构建请求字典
    request = {
        'id': req_id,
        'source': source,
        'dest': dest,  # 列表形式的目的节点
        'vnf': vnf_types,
        'cpu_origin': cpu_origin,
        'memory_origin': memory_origin,
        'bw_origin': bw_origin,
        'arrive_time': arrive_time,
        'lifetime': lifetime,
        'leave_time': leave_time,
        'arrive_time_step': math.ceil(arrive_time),
        'leave_time_step': math.ceil(leave_time)
    }
    return request


def generate_node_requests(source, node_important, arrive_time_list, all_vnf):
    """
    为特定源节点生成一系列请求
    """
    node_requests = []

    # 候选目的节点：除了源节点以外的重要节点
    candidates = [n for n in node_important if n != source]

    # 对应图片中的参数：
    max_bandwidth = 8  # 业务请求带宽资源需求量上限
    min_bandwidth = 4  # 业务请求带宽资源需求量下限
    multicast_num = 5  # 业务请求目的节点个数 5
    mean_lifetime = 3  # 指定分布均值

    for i, arrive_time in enumerate(arrive_time_list):
        # 随机选择多播目的节点
        k = min(multicast_num, len(candidates))
        dest = random.sample(candidates, k)

        # ID 暂时设为0，后续统一重排
        req = generate_single_request(0, source, dest, all_vnf,
                                      max_bandwidth, min_bandwidth,
                                      arrive_time, mean_lifetime)
        node_requests.append(req)

    return node_requests