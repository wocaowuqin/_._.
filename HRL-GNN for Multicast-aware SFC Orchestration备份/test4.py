import pickle
from collections import Counter
from pathlib import Path


REQ_PATH = r"/data/input_dir/generate_requests_depend_on_poisson/data_output/phase1_requests.pkl"
EVT_PATH = r"E:\pycharmworkspace\SFC-master\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\phase1_events.pkl"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_requests(reqs):
    print("\n" + "=" * 80)
    print("📦 ANALYZE phase1_requests.pkl")
    print("=" * 80)

    print(f"Total requests: {len(reqs)}")

    ids = []
    for i, r in enumerate(reqs):
        rid = r.get("id", None)
        ids.append(rid)

        if i < 3:
            print(f"\nSample request[{i}]:")
            print(f"  id    = {rid}")
            print(f"  src   = {r.get('src')}")
            print(f"  dests = {r.get('dests')}")
            print(f"  vnfs  = {r.get('vnfs')}")

    counter = Counter(ids)

    print("\n--- ID Statistics ---")
    print(f"Unique IDs: {len(counter)}")
    print(f"Most common IDs: {counter.most_common(5)}")

    if len(counter) == 1:
        print("❌ ALL requests have the SAME id!")
    else:
        print("✅ Request IDs vary")

    dup_ids = [rid for rid, c in counter.items() if c > 1]
    print(f"Duplicate IDs count: {len(dup_ids)}")

    if dup_ids:
        print(f"Example duplicate id: {dup_ids[0]}")


def analyze_events(evts):
    print("\n" + "=" * 80)
    print("📦 ANALYZE phase1_events.pkl")
    print("=" * 80)

    print(f"Total events: {len(evts)}")

    ids = []
    for i, e in enumerate(evts):
        rid = e.get("req_id", e.get("id", None))
        ids.append(rid)

        if i < 3:
            print(f"\nSample event[{i}]:")
            print(e)

    counter = Counter(ids)

    print("\n--- Event Request-ID Statistics ---")
    print(f"Unique request IDs in events: {len(counter)}")
    print(f"Most common IDs: {counter.most_common(5)}")

    if len(counter) == 1:
        print("❌ ALL events reference the SAME request id!")
    else:
        print("✅ Event request IDs vary")


def cross_check(reqs, evts):
    print("\n" + "=" * 80)
    print("🔍 CROSS CHECK requests ↔ events")
    print("=" * 80)

    req_ids = set(r.get("id") for r in reqs)
    evt_ids = set(e.get("req_id", e.get("id")) for e in evts)

    missing = evt_ids - req_ids

    print(f"Request IDs in requests: {len(req_ids)}")
    print(f"Request IDs in events:   {len(evt_ids)}")

    if missing:
        print(f"❌ Event refers to missing request IDs: {list(missing)[:5]}")
    else:
        print("✅ All event request IDs exist in requests")


def main():
    print("🚀 Loading datasets...")
    requests = load_pkl(REQ_PATH)
    events = load_pkl(EVT_PATH)

    analyze_requests(requests)
    analyze_events(events)
    cross_check(requests, events)

    print("\n✅ Dataset check finished.")


if __name__ == "__main__":
    main()
