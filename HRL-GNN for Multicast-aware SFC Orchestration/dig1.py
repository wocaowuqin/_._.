"""
目的节点距离分析
检查请求中目的节点是否聚集还是分散
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dest_clustering(env, num_samples=100):
    """
    分析目的节点的聚集程度
    """
    print("=" * 60)
    print("目的节点距离分析")
    print("=" * 60)

    results = []

    for i in range(num_samples):
        env.reset()
        req = env.current_request

        if req is None:
            continue

        src = req.get('source', 0)
        dests = req.get('dest', [])

        if len(dests) < 2:
            continue

        # 计算所有目的节点之间的距离
        inter_dest_distances = []
        for j, d1 in enumerate(dests):
            for d2 in dests[j+1:]:
                path = env._find_path(d1, d2)
                if path:
                    dist = len(path) - 1
                    inter_dest_distances.append(dist)

        # 计算源节点到每个目的节点的距离
        src_to_dest_distances = []
        for d in dests:
            path = env._find_path(src, d)
            if path:
                dist = len(path) - 1
                src_to_dest_distances.append(dist)

        if not inter_dest_distances or not src_to_dest_distances:
            continue

        result = {
            'req_id': req.get('request_id', i),
            'src': src,
            'dests': dests,
            'num_dests': len(dests),
            'avg_inter_dest_dist': np.mean(inter_dest_distances),
            'max_inter_dest_dist': np.max(inter_dest_distances),
            'min_inter_dest_dist': np.min(inter_dest_distances),
            'std_inter_dest_dist': np.std(inter_dest_distances),
            'avg_src_to_dest': np.mean(src_to_dest_distances),
            'max_src_to_dest': np.max(src_to_dest_distances),
            'min_src_to_dest': np.min(src_to_dest_distances),
        }

        results.append(result)

    # 统计分析
    print(f"\n分析了 {len(results)} 个请求")
    print("\n【目的节点间距离统计】")
    avg_inter = [r['avg_inter_dest_dist'] for r in results]
    max_inter = [r['max_inter_dest_dist'] for r in results]

    print(f"  平均距离: {np.mean(avg_inter):.2f} ± {np.std(avg_inter):.2f}")
    print(f"  最大距离: {np.mean(max_inter):.2f} ± {np.std(max_inter):.2f}")

    print("\n【源到目的节点距离统计】")
    avg_src = [r['avg_src_to_dest'] for r in results]
    max_src = [r['max_src_to_dest'] for r in results]

    print(f"  平均距离: {np.mean(avg_src):.2f} ± {np.std(avg_src):.2f}")
    print(f"  最大距离: {np.mean(max_src):.2f} ± {np.std(max_src):.2f}")

    # 聚集度分析
    print("\n【聚集度分析】")
    clustered = 0  # 聚集型（目的节点彼此靠近）
    dispersed = 0  # 分散型（目的节点分散）

    for r in results:
        if r['avg_inter_dest_dist'] < 3.0:
            clustered += 1
        elif r['avg_inter_dest_dist'] > 6.0:
            dispersed += 1

    print(f"  聚集型 (平均间距<3): {clustered} ({clustered/len(results)*100:.1f}%)")
    print(f"  分散型 (平均间距>6): {dispersed} ({dispersed/len(results)*100:.1f}%)")
    print(f"  中等型: {len(results)-clustered-dispersed} ({(len(results)-clustered-dispersed)/len(results)*100:.1f}%)")

    # 找出最分散的请求
    print("\n【最分散的5个请求】")
    sorted_results = sorted(results, key=lambda x: x['max_inter_dest_dist'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. Req={r['req_id']}, Src={r['src']}, Dests={r['dests']}")
        print(f"     目的节点间最大距离: {r['max_inter_dest_dist']}")
        print(f"     源到目的最大距离: {r['max_src_to_dest']}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：目的节点间距离分布
    axes[0].hist(avg_inter, bins=20, alpha=0.7, label='平均间距')
    axes[0].hist(max_inter, bins=20, alpha=0.7, label='最大间距')
    axes[0].axvline(3.0, color='g', linestyle='--', label='聚集阈值')
    axes[0].axvline(6.0, color='r', linestyle='--', label='分散阈值')
    axes[0].set_xlabel('距离 (跳数)')
    axes[0].set_ylabel('请求数量')
    axes[0].set_title('目的节点间距离分布')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 图2：目的节点数量 vs 平均间距
    num_dests = [r['num_dests'] for r in results]
    axes[1].scatter(num_dests, avg_inter, alpha=0.5)
    axes[1].set_xlabel('目的节点数量')
    axes[1].set_ylabel('平均节点间距')
    axes[1].set_title('节点数量 vs 聚集度')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_dir = Path('outputs/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'dest_clustering_analysis.png', dpi=150)
    print(f"\n📊 图表已保存到: {output_dir / 'dest_clustering_analysis.png'}")

    return results

# 使用示例
if __name__ == "__main__":
    import sys
    sys.path.append('.')

    from utils.config_utils import load_config
    from envs.sfc_env import SFC_HIRL_Env

    config = load_config('phase3')
    env = SFC_HIRL_Env(config, use_gnn=True)

    results = analyze_dest_clustering(env, num_samples=200)

    print("\n" + "=" * 60)
    print("✅ 分析完成")
    print("=" * 60)