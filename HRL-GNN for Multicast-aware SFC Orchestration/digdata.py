import pickle
import os

# 替换成你实际的数据路径
data_path = "data/input_dir/phase3_requests.pkl"

if os.path.exists(data_path):
    with open(data_path, 'rb') as f:
        requests = pickle.load(f)

    if len(requests) > 0:
        print("🔍 数据集字段一览：")
        print(requests[0].keys())  # 看看第一个请求有哪些 Key

        # 检查有没有类似的时间字段
        sample = requests[0]
        print("\n样本数据：")
        for k in ['ttl', 'duration', 'life_time', 'holding_time', 'time']:
            if k in sample:
                print(f"✅ 发现字段 '{k}': {sample[k]}")
            else:
                print(f"❌ 未发现字段 '{k}'")
    else:
        print("数据集为空")
else:
    print("文件不存在")