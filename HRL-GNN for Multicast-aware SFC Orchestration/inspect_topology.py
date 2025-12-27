import scipy.io
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_path_from_struct(cell):
    """
    [V4] 专门针对 struct 结构提取路径
    """
    try:
        # 1. 检查是否为结构化数组 (即包含字段名)
        if cell.dtype.names and 'paths' in cell.dtype.names:
            # 提取 'paths' 字段
            paths_data = cell['paths']

            # paths_data 通常是一个 object array，里面包着真正的矩阵
            if isinstance(paths_data, np.ndarray) and paths_data.size > 0:
                # 取出内容 (解包 object)
                real_paths = paths_data.item() if paths_data.ndim == 0 else paths_data[0]

                # 如果 real_paths 还是包着一层 (比如 (1, 1) 的 cell)
                if isinstance(real_paths, np.ndarray) and real_paths.dtype == 'O':
                    if real_paths.size > 0:
                        real_paths = real_paths[0]

                # 现在 real_paths 应该是那个 uint8 的二维数组了
                # [[1, 2, 0...], [1, 4, 7...]]
                if isinstance(real_paths, np.ndarray):
                    # 取第一行 (最短路径)
                    shortest_path = real_paths[0] if real_paths.ndim > 1 else real_paths

                    # 过滤掉 0 (Padding)
                    clean_path = [x for x in shortest_path.flatten() if x != 0]
                    return clean_path

    except Exception as e:
        # print(f"解析出错: {e}")
        pass

    return []


def inspect_mat_paths_v4(file_path):
    print(f"🔍 [V4] 正在读取文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在! {file_path}")
        return

    try:
        mat_data = scipy.io.loadmat(file_path)
        paths_matrix = mat_data['Paths']
        print(f"✅ 成功加载 Paths 矩阵，形状: {paths_matrix.shape}")

        G = nx.Graph()
        rows, cols = paths_matrix.shape
        valid_count = 0

        print("⏳ 正在精确解析结构化路径...")

        # 遍历矩阵
        for i in range(rows):
            for j in range(i + 1, cols):
                cell = paths_matrix[i, j]

                # 提取路径
                path = extract_path_from_struct(cell)

                if path and len(path) > 1:
                    valid_count += 1
                    # 路径中的数字通常是 1-based (MATLAB习惯)
                    # 我们先直接添加，画图时再确认
                    for k in range(len(path) - 1):
                        u = int(path[k])
                        v = int(path[k + 1])
                        G.add_edge(u, v)

        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            print("❌ 解析失败，依然没有提取到节点。")
            return

        # 统计
        print(f"📊 还原成功!")
        print(f"   - 提取到有效路径数据: {valid_count} 条")
        print(f"   - 节点数: {num_nodes} (ID范围: {min(G.nodes)} ~ {max(G.nodes)})")
        print(f"   - 边数:   {G.number_of_edges()}")

        # 🧐 关键节点对比 (核心环节)
        print("-" * 40)
        print("🧐 关键节点连接情况 (请与教科书对比):")

        # 判断是 0-based 还是 1-based
        # 根据日志 Raw Content: array([[1, 2, 0...]])，看起来是 1-based
        # 所以 Node 1 就是图上的 Node 1

        check_ids = [1, 2, 3, 28]  # 检查这几个关键点
        for nid in check_ids:
            if nid in G:
                nbrs = sorted(list(G.neighbors(nid)))
                print(f"   - Node {nid} 连接到: {nbrs}")
            else:
                print(f"   - Node {nid} 不在图中")
        print("-" * 40)

        # 绘图
        plt.figure(figsize=(12, 10))
        # 使用力导向布局，但尝试调整参数让它展开一点
        pos = nx.spring_layout(G, seed=42, k=0.6, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title(f"Reconstructed Topology (V4)\nNodes: {num_nodes} | Edges: {G.number_of_edges()}")
        plt.axis('off')
        plt.savefig("topology_v4_check.png")
        print("🖼️  拓扑检查图已保存: topology_v4_check.png")
        plt.show()

    except Exception as e:
        print(f"❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    target_file = "data/input_dir/US_Backbone_path.mat"
    if not os.path.exists(target_file):
        target_file = "US_Backbone_path.mat"

    inspect_mat_paths_v4(target_file)