import scipy.io
import numpy as np
import os

# 你的文件路径
TOPO_FILE = r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\US_Backbone_path.mat"


def inspect_mat_file():
    print("=" * 60)
    print(f"🕵️‍♂️ 正在诊断拓扑文件: {os.path.basename(TOPO_FILE)}")
    print("=" * 60)

    if not os.path.exists(TOPO_FILE):
        print(f"❌ 错误: 找不到文件 \n{TOPO_FILE}")
        return

    try:
        # 加载 .mat 文件
        mat_data = scipy.io.loadmat(TOPO_FILE)

        # 遍历里面的所有变量
        found_data = False
        for key, val in mat_data.items():
            if key.startswith('__'): continue  # 跳过元数据

            found_data = True
            print(f"\n🔍 发现变量: '{key}'")
            print(f"   类型: {type(val)}")

            if isinstance(val, np.ndarray):
                print(f"   形状: {val.shape}")

                # 检查是否包含数值
                if np.issubdtype(val.dtype, np.number):
                    min_val = np.min(val)
                    max_val = np.max(val)
                    print(f"   📉 数值范围: [{min_val}, {max_val}]")

                    # 核心判断逻辑
                    if min_val == 1 and max_val == 28:
                        print("   ✅ 判定: 1-based (MATLAB 风格) [1, 28]")
                    elif min_val == 0 and max_val == 27:
                        print("   ✅ 判定: 0-based (Python 风格) [0, 27]")
                    else:
                        print("   ⚠️ 判定: 数值范围不典型，请人工确认")

                # 如果是 cell array (通常存路径列表)
                elif val.dtype == 'O':
                    print("   📦 检测到对象数组 (可能是路径列表)")
                    try:
                        # 尝试抽取第一个非空元素的第一个值来看看
                        sample = val.flat[0]
                        if isinstance(sample, np.ndarray) and sample.size > 0:
                            print(f"   👀 样本内容: {sample.flatten()}")
                            print(f"   📉 样本范围: [{np.min(sample)}, {np.max(sample)}]")
                    except:
                        pass
            else:
                print(f"   内容: {val}")

    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    inspect_mat_file()