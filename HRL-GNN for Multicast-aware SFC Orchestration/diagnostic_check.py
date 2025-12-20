import pickle
import numpy as np
import networkx as nx
from core.expert.expert_msfce.core.resource_manager import ResourceManager  # 确保能导入你的 RM 类

# ================= 配置区 =================
# 你的拓扑文件路径 (如果有的话，或者直接用代码生成)
# 这里模拟你的 RM 初始化参数，请根据你的 main.py 实际情况调整
NUM_NODES = 28
# 假设这是你的请求数据路径
REQUEST_DATA_PATH = 'data/input_dir/phase1_requests.pkl'


# ==========================================

class IDDiagnostic:
    def __init__(self):
        print("=" * 60)
        print("🕵️‍♂️ 正在启动 ID 诊断程序...")
        print("=" * 60)

    def load_requests(self):
        try:
            with open(REQUEST_DATA_PATH, 'rb') as f:
                self.requests = pickle.load(f)
            print(f"✅ 成功加载请求文件: {len(self.requests)} 条记录")
        except FileNotFoundError:
            print(f"❌ 找不到文件: {REQUEST_DATA_PATH}")
            print("   -> 请生成数据后再运行诊断，或者修改脚本中的路径。")
            self.requests = []

    def inspect_data_range(self):
        if not self.requests: return

        print("\n[1] 检查请求数据 (External Data):")

        sources = [r['source'] for r in self.requests]
        all_dests = []
        for r in self.requests:
            all_dests.extend(r['dest'] if isinstance(r['dest'], list) else [r['dest']])

        min_s, max_s = min(sources), max(sources)
        min_d, max_d = min(all_dests), max(all_dests)

        print(f"   - Source ID 范围: [{min_s}, {max_s}]")
        print(f"   - Dest ID 范围:   [{min_d}, {max_d}]")

        # 重点检查节点 28
        count_28_src = sources.count(28)
        count_28_dest = all_dests.count(28)
        print(f"   - ⚠️ 节点 28 出现次数: 作为源={count_28_src}, 作为目的={count_28_dest}")

        if max_s > NUM_NODES - 1:
            print(f"   🚨 发现潜在风险: 最大 ID ({max_s}) 超过了 0-based 索引上限 ({NUM_NODES - 1})")

    def inspect_rm_internal(self):
        print("\n[2] 检查 ResourceManager 内部状态 (Internal State):")

        # 模拟初始化 RM (使用你提供的巨大资源参数)
        # 注意：这里我们使用默认初始化，不传 node_index_base，看看默认行为
        topo = np.ones((NUM_NODES, NUM_NODES))  # 全连接虚拟拓扑用于测试
        caps = {'cpu': 800, 'memory': 600, 'bandwidth': 1200}
        dc_nodes = [1, 2, 3]  # 随意填，不影响索引检查

        try:
            rm = ResourceManager(topo, caps, dc_nodes)
            print(f"   ✅ RM 初始化成功。")
            print(f"   - rm.n (节点总数): {rm.n}")
            print(f"   - rm.C 数组大小: {len(rm.C)}")
            print(f"   - rm.C 索引范围: [0, {len(rm.C) - 1}]")

            # 检查内部 Graph 构建逻辑
            # 我们手动构建一个只有一条边的图，看看 RM 把它存成了什么
            rm.link_map = {(1, 2): 1}  # 假设有一条边连接节点 1 和 2
            # 这里的 1 和 2 如果是外部 ID，RM 内部是原样存的吗？

            print(f"   - 内部 Graph 节点检查: 假设拓扑包含 28 个节点")
            if rm.n == 28:
                print(f"     -> 内部最大合法索引是: 27")

            return rm
        except Exception as e:
            print(f"   ❌ RM 初始化失败: {e}")
            return None

    def simulate_conflict(self, rm):
        if not rm or not self.requests: return

        print("\n[3] 模拟冲突场景 (Simulation):")

        # 找一个包含节点 28 的请求
        problem_req = next((r for r in self.requests if r['source'] == 28), None)

        if problem_req:
            src = problem_req['source']
            print(f"   提取请求 ID {problem_req['id']}: Source = {src}")

            # 模拟 check_global_feasibility 中的关键判断
            # G_bw = self._build_bw_feasible_subgraph(bw_req)
            # if not G_bw.has_node(source): ...

            # 我们直接看 rm.C (CPU数组) 能不能取到这个值
            try:
                # 尝试访问内部数组
                _ = rm.C[src]
                print(f"   ✅ rm.C[{src}] 访问成功。")
            except IndexError:
                print(f"   ❌ rm.C[{src}] 访问失败! Index out of bounds.")
                print(f"   💥 实锤原因: 请求传入了 ID {src}，但 RM 内部数组最大下标只有 {rm.n - 1}。")
                print(f"   📉 这导致任何涉及节点 {src} 的请求直接报错/不可行。")

        else:
            print("   (样本中未找到源节点为 28 的请求，跳过模拟)")


if __name__ == "__main__":
    diag = IDDiagnostic()
    diag.load_requests()
    rm = diag.inspect_rm_internal()
    diag.simulate_conflict(rm)