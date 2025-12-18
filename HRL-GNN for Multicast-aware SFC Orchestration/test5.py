import pickle
import numpy as np
from collections import Counter
from torch_geometric.data import Data

EXPERT_PATH = r"E:/pycharmworkspace/SFC-master/HRL-GNN for Multicast-aware SFC Orchestration/outputs/expert/expert_data_final.pkl"

print("🚀 Loading expert data...")
with open(EXPERT_PATH, "rb") as f:
    data = pickle.load(f)

print("=" * 80)
print("📦 BASIC STATISTICS")
print("=" * 80)

print(f"Total samples: {len(data)}")

# -------------------------
# 1️⃣ State / Action 完整性
# -------------------------
missing_state = 0
missing_action = 0
not_graph_state = 0

for s in data:
    state = s.get("state") or s.get("network_state")
    action = s.get("action")

    if state is None:
        missing_state += 1
    if action is None:
        missing_action += 1
    if state is not None and not isinstance(state, Data):
        not_graph_state += 1

print(f"Missing state:  {missing_state}")
print(f"Missing action: {missing_action}")
print(f"Non-graph states: {not_graph_state}")

# -------------------------
# 2️⃣ Graph 字段检查
# -------------------------
missing_fields = Counter()

for s in data:
    state = s.get("state") or s.get("network_state")
    if state is None or not isinstance(state, Data):
        continue

    for k in ["x", "edge_index", "edge_attr"]:
        if not hasattr(state, k):
            missing_fields[k] += 1

    if not (hasattr(state, "req_vec") or hasattr(state, "req")):
        missing_fields["req_vec"] += 1

print("\n🧠 Graph field missing counts:")
for k, v in missing_fields.items():
    print(f"  {k}: {v}")

# -------------------------
# 3️⃣ Action 分布
# -------------------------
actions = [s["action"] for s in data if "action" in s]
action_counter = Counter(actions)

print("\n🎯 ACTION DISTRIBUTION (top 10):")
for a, c in action_counter.most_common(10):
    print(f"  Action {a}: {c} ({c / len(actions) * 100:.2f}%)")

# -------------------------
# 4️⃣ Success / Failure 分布
# -------------------------
success_flags = []
for s in data:
    if "success" in s:
        success_flags.append(s["success"])

if success_flags:
    succ_rate = sum(success_flags) / len(success_flags)
    print(f"\n✅ Success rate: {succ_rate * 100:.2f}%")
else:
    print("\n⚠️ No success flag found in data")

# -------------------------
# 5️⃣ 状态重复性（冲突风险）
# -------------------------
print("\n🔍 STATE → ACTION CONSISTENCY CHECK")

state_hash = {}
conflict = 0

for s in data:
    state = s.get("state") or s.get("network_state")
    action = s.get("action")
    if state is None or action is None:
        continue

    h = tuple(state.x.view(-1).tolist())  # 粗略 hash
    if h in state_hash and state_hash[h] != action:
        conflict += 1
    else:
        state_hash[h] = action

print(f"Conflicting state-action pairs: {conflict}")

print("\n✅ Expert data inspection finished.")
