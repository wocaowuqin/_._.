import scipy.io
import numpy as np
import os


def verify_mat_indexing():
    # ⚠️ 请确保此处路径正确
    mat_path = r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\US_Backbone_path.mat"

    if not os.path.exists(mat_path):
        print(f"❌ 文件不存在: {mat_path}")
        return

    print(f"📂 正在加载: {mat_path}")
    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    if 'Paths' not in mat_data:
        print("❌ 变量 'Paths' 不在文件中")
        return

    paths_matrix = mat_data['Paths']
    rows, cols = paths_matrix.shape
    print(f"✅ Paths 矩阵维度: {rows} x {cols}")

    # 用于统计的变量
    all_values = set()
    path_count = 0
    empty_count = 0

    # 遍历矩阵
    for r in range(rows):
        for c in range(cols):
            if r == c: continue  # 跳过对角线

            cell = paths_matrix[r, c]

            # 检查是否有 paths 字段
            if cell.dtype.names is None or 'paths' not in cell.dtype.names:
                continue

            raw_paths = cell['paths']

            if raw_paths.size == 0:
                empty_count += 1
                continue

            # 提取路径数据
            # 处理可能的嵌套 object array 或直接的 numpy array
            try:
                # 将数据展平以便统一处理
                flat_data = raw_paths.flatten()

                for item in flat_data:
                    # 如果 item 还是数组（例如多条路径），继续展平
                    if isinstance(item, np.ndarray):
                        nodes = item.flatten()
                    else:
                        nodes = np.array([item])

                    # 转换为整数列表
                    node_list = [int(x) for x in nodes]

                    if len(node_list) > 0:
                        path_count += 1
                        for node_id in node_list:
                            all_values.add(node_id)

            except Exception as e:
                pass

    # 统计分析
    if not all_values:
        print("❌ 未提取到任何节点数据")
        return

    min_val = min(all_values)
    max_val = max(all_values)
    has_zero = 0 in all_values
    has_one = 1 in all_values

    print("\n" + "=" * 30)
    print("📊 索引验证报告")
    print("=" * 30)
    print(f"检查的路径片段数: {path_count}")
    print(f"节点 ID 最小值: {min_val}")
    print(f"节点 ID 最大值: {max_val}")
    print(f"包含 0: {'是' if has_zero else '否'}")
    print(f"包含 1: {'是' if has_one else '否'}")
    print("-" * 30)

    # 智能推断结论
    if min_val == 1:
        print("✅ 结论: 肯定是 1-based 索引")
        print("   (节点从 1 开始，符合 MATLAB 生成习惯)")
        print("   👉 您的 Python 代码需要将 Env 的 0-based 请求 +1 才能匹配")

    elif min_val == 0:
        print("⚠️ 结论: 包含 0，需进一步判断")
        if has_one:
            # 如果既有0又有1，需要判断0是否仅作为填充(Padding)
            # 在 expert_msfce.py 中有一行：path_nodes = [int(x) for x in path_segment if int(x) > 0]
            # 这暗示 0 被视为无效填充。
            print("   虽然包含 0，但通常 MATLAB 导出的矩阵会用 0 做补齐(Padding)。")
            print("   如果您的 Expert 代码逻辑中有 `if x > 0` 的过滤，则实际有效节点仍是从 1 开始。")
            print("   👉 建议：仍然按照 1-based 对待，并确保代码过滤掉了 0。")
        else:
            print("   数据可能是 0-based，或者非常稀疏。")

    else:
        print(f"❓ 结论: 索引范围异常 ({min_val} - {max_val})")


if __name__ == "__main__":
    verify_mat_indexing()