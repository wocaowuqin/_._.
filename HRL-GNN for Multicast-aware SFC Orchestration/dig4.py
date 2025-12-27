def verify_dc_nodes():
    print("🔍 正在核对 Phase 3 数据转换逻辑...")
    print("=" * 50)

    # 1. 您提供的真实定义 (1-based, 对应 MATLAB 或文档)
    user_dc_ids = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]
    print(f"📋 用户定义的 DC ID (1-based):\n   {user_dc_ids}")
    print(f"   (共 {len(user_dc_ids)} 个节点)")

    # 2. 模拟代码中的转换 (转为 0-based 索引)
    # Python 索引 = ID - 1
    code_dc_indices = set([x - 1 for x in user_dc_ids])
    sorted_indices = sorted(list(code_dc_indices))

    print("-" * 50)
    print(f"💻 代码内部应使用的索引 (0-based Index):\n   {sorted_indices}")

    # 3. 💣 核心冲突检测 (基于您之前的日志)
    # 之前的日志显示：Agent 部署在了 Node 8 (Index 8) 和 Node 3 (Index 3)

    print("-" * 50)
    print("🕵️‍♂️ 关键节点「合法性」验尸:")

    # 检查 VNF@3 (Index 3) -> 对应 ID 4
    check_idx_3 = 3
    is_3_ok = check_idx_3 in code_dc_indices
    print(f"   [Index 3] (对应 ID 4): {'✅ 合法 DC' if is_3_ok else '❌ 非法 (纯转发节点)'}")

    # 检查 VNF@5 (Index 5) -> 对应 ID 6
    check_idx_5 = 5
    is_5_ok = check_idx_5 in code_dc_indices
    print(f"   [Index 5] (对应 ID 6): {'✅ 合法 DC' if is_5_ok else '❌ 非法 (纯转发节点)'}")

    # 🔥 检查 VNF@8 (Index 8) -> 对应 ID 9
    check_idx_8 = 8
    is_8_ok = check_idx_8 in code_dc_indices
    print(f"   [Index 8] (对应 ID 9): {'✅ 合法 DC' if is_8_ok else '❌ 违规! (在非DC节点部署)'}")

    print("=" * 50)

    # 4. 最终判决
    if not is_8_ok:
        print("🚨 【严重警告】数据有问题！")
        print("   原因: 您刚才的日志显示 Agent 在 [Index 8] 部署了 VNF。")
        print("   事实: 根据您的列表，[Index 8] 对应 [ID 9]。")
        print("   列表: 您的列表从 8 跳到了 11，中间没有 9！")
        print("   -> 结论: 之前的代码并没有真正使用您提供的这个列表，而是使用了默认的(可能是全节点)配置。")
        print("   -> 建议: 必须去 sfc_env.py 手动更新 self.dc_nodes！")
    else:
        print("✅ 数据转换没问题，配置一致。")


if __name__ == "__main__":
    verify_dc_nodes()