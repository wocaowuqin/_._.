# debug_paths.py
import scipy.io
import numpy as np
import os


def debug_paths_structure():
    # 路径根据您的报错日志调整
    mat_path = r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\US_Backbone_path.mat"

    if not os.path.exists(mat_path):
        print(f"❌ 文件不存在: {mat_path}")
        return

    print(f"📂 加载文件: {mat_path}")
    mat_data = scipy.io.loadmat(mat_path)

    if 'Paths' not in mat_data:
        print("❌ 未找到 'Paths' 变量")
        return

    paths_matrix = mat_data['Paths']
    print(f"✅ Paths 形状: {paths_matrix.shape}")

    # 获取第0行第1列的数据（节点0 -> 节点1）
    # 找一个非对角线的元素
    row, col = 0, 1
    cell = paths_matrix[row, col]

    print(f"\n🔍 正在检查 Paths[{row}, {col}] 的内容...")
    print(f"类型: {type(cell)}")
    print(f"Dtype: {cell.dtype}")

    # 尝试访问 'paths' 字段
    try:
        paths_field = cell['paths']
        print(f"\n字段 'paths' 获取成功!")
        print(f"类型: {type(paths_field)}")
        print(f"Shape: {paths_field.shape}")
        print(f"内容摘要: {paths_field}")

        # 深度解包测试
        if paths_field.size > 0:
            item0 = paths_field[0]
            print(f"\nitem[0] 类型: {type(item0)}")
            print(f"item[0] 内容: {item0}")

            if hasattr(item0, '__len__') and len(item0) > 0:
                print(f"item[0][0] 内容: {item0[0]}")

    except Exception as e:
        print(f"❌ 访问 'paths' 失败: {e}")


if __name__ == "__main__":
    debug_paths_structure()