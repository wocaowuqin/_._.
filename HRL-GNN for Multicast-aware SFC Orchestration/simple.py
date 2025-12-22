#!/usr/bin/env python3
"""简单的 GNN 输出测试"""

import torch
from utils.config_utils import load_config

print("=" * 70)
print("🧪 简单 GNN 测试")
print("=" * 70)

# 加载配置
config = load_config('phase2')

# 导入 GNN
from core.gnn.multicast_aware_gat import MulticastAwareGAT

# 初始化 GNN
print("\n1. 初始化 GNN...")
try:
    gnn = MulticastAwareGAT(
        node_feat_dim=32,
        edge_feat_dim=16,
        request_dim=6,
        hidden_dim=128,
        # 不传 action_dim，看看会怎样
        num_layers=3,
        num_heads=4,
        dropout=0.0
    )
    print("✅ GNN 初始化成功（不传 action_dim）")
    print(f"   GNN mode: {gnn.mode}")
    print(f"   GNN output_layer: {gnn.output_layer}")

    if gnn.output_layer is None:
        print("   ✅ output_layer 是 None（正确！）")
    else:
        print("   ❌ output_layer 不是 None（错误！）")
        print(f"      {gnn.output_layer}")

except Exception as e:
    print(f"❌ GNN 初始化失败: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# 创建假数据
print("\n2. 创建测试数据...")
num_nodes = 28
num_edges = 100

x = torch.randn(num_nodes, 32)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.randn(num_edges, 16)
req_vec = torch.randn(6)
batch = torch.zeros(num_nodes, dtype=torch.long)

print(f"✅ 测试数据创建成功")
print(f"   x: {x.shape}")
print(f"   edge_index: {edge_index.shape}")
print(f"   edge_attr: {edge_attr.shape}")
print(f"   req_vec: {req_vec.shape}")

# 前向传播
print("\n3. 测试前向传播...")
try:
    with torch.no_grad():
        output = gnn(x, edge_index, edge_attr, req_vec, batch)

    print(f"✅ 前向传播成功")
    print(f"   输出形状: {output.shape}")
    print(f"   期望形状: (1, 128)")

    if output.shape == torch.Size([1, 128]):
        print(f"   ✅ 输出维度正确！")
    else:
        print(f"   ❌ 输出维度错误！")

    # 检查输出统计
    print(f"\n   输出统计:")
    print(f"     最小值: {output.min().item():.4f}")
    print(f"     最大值: {output.max().item():.4f}")
    print(f"     均值: {output.mean().item():.4f}")
    print(f"     标准差: {output.std().item():.4f}")

    if output.std().item() < 0.01:
        print(f"   ❌ 输出几乎是常数（std < 0.01）")
    else:
        print(f"   ✅ 输出正常变化")

except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("🎯 结论")
print("=" * 70)
print("""
如果看到：
  ✅ output_layer 是 None
  ✅ 输出维度正确 (1, 128)
  ✅ 输出正常变化

说明 GNN 修复成功！可以重新训练。

如果看到：
  ❌ output_layer 不是 None
  ❌ 输出维度错误

说明 multicast_aware_gat.py 没有正确替换！
需要手动检查文件内容。
""")
print("=" * 70)