# trainer/phase1_collector.py
import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    def __init__(self, env, expert_solver, output_dir: str, max_episodes: int = 5000, save_every: int = 500):
        self.env = env
        self.expert = expert_solver
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_requests = max_episodes
        self.save_every = save_every
        self.success_samples = []
        self.fail_contexts = []
        self.stats = {"requests": 0, "success": 0, "fail": 0}

    def _estimate_load(self):
        rm = self.env.resource_mgr
        bw_util = 1.0 - rm.B.mean() / max(1.0, rm.B_cap)
        cpu_util = 1.0 - rm.C.mean() / max(1.0, rm.C_cap)
        return 0.5 * bw_util + 0.5 * cpu_util

    def collect(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection (Fixed Version)")
        self.env.reset(options={'phase': 'phase1'})

        if hasattr(self.env, 'data_loader'):
            events = self.env.data_loader.events
            requests = self.env.data_loader.requests
        else:
            events = getattr(self.env, 'events', [])
            requests = getattr(self.env, 'requests', [])

        if not requests:
            logger.error("❌ No requests found!")
            return

        pbar = tqdm(total=len(events), desc="Collecting", ncols=120)

        for t, event in enumerate(events):
            pbar.update(1)

            # 处理离开
            arrive_list = event.get("arrive", event.get("arrive_event", []))
            leave_list = event.get("leave", event.get("leave_event", []))

            for leave_req_id in leave_list:
                try:
                    self.env.event_handler.unregister_service(leave_req_id)
                except:
                    pass

            for req_id in arrive_list:
                if req_id <= 0 or req_id > len(requests): continue
                req = requests[req_id - 1]
                self.stats["requests"] += 1

                # 1. 捕获 State
                self.env.current_request = req
                dummy_tree = {'tree': {}, 'hvt': np.zeros((self.env.n, 8))}
                x, edge_index, edge_attr, req_vec = self.env.resource_mgr.get_graph_state(
                    current_request=req, nodes_on_tree={req['source']},
                    current_tree=dummy_tree, served_dest_count=0, sharing_strategy=0, nb_high_goals=10
                )
                state_to_save = Data(x=x.cpu(), edge_index=edge_index.cpu(), edge_attr=edge_attr.cpu(),
                                     req_vec=req_vec.cpu())

                # 2. Expert Solve
                network_state = self.env.resource_mgr.get_network_state_dict(req)
                expert_result = self.expert.solve_request_for_expert(req, network_state)

                success = False
                if expert_result is not None:
                    tree_data, traj = expert_result

                    if tree_data is not None:
                        # 🔥【关键修复】将 Array Tree 转换为 Dict Tree
                        # ResourceManager 需要 {(u,v): 1.0} 格式
                        dict_tree = {}
                        paths_map = tree_data.get('paths_map', {})
                        for dest, path_nodes in paths_map.items():
                            # path_nodes 可能是 1-based, 转 0-based
                            path_0 = [n - 1 if n > 0 else 0 for n in path_nodes]
                            for i in range(len(path_0) - 1):
                                u, v = path_0[i], path_0[i + 1]
                                dict_tree[(u, v)] = 1.0
                                dict_tree[(v, u)] = 1.0  # 双向占用

                        # 构造最终 plan
                        deployment_plan = {
                            'hvt': tree_data['hvt'],  # (N, K) array
                            'tree': dict_tree  # Dict {(u,v): 1}
                        }

                        # 尝试部署
                        if self.env.resource_mgr.apply_tree_deployment(deployment_plan, req):
                            success = True
                            self.stats["success"] += 1
                            # 注册到 handler (用于释放)
                            self.env.event_handler.services[req_id] = {
                                'req': req,
                                'tree': deployment_plan,  # 存 dict tree
                                'hvt': tree_data['hvt']
                            }

                            for (dest_idx, action, cost) in traj:
                                self.success_samples.append({
                                    "state": state_to_save,
                                    "request": req,
                                    "high_action": dest_idx,
                                    "action": action,
                                    "cost": cost,
                                    "load": self._estimate_load(),
                                })
                        else:
                            # 部署被 Env 拒绝 (资源不足?)
                            pass

                if not success:
                    self.stats["fail"] += 1
                    self.fail_contexts.append({"request": req, "reason": "expert_failed"})

                br = self.stats["fail"] / max(1, self.stats["requests"])
                pbar.set_postfix(succ=self.stats["success"], fail=self.stats["fail"], BR=f"{br:.2%}")

            if self.stats["requests"] >= self.max_requests: break
            if self.stats["requests"] > 0 and self.stats["requests"] % self.save_every == 0:
                self._save_final()

        pbar.close()
        self._save_final()
        return self.stats

    def _save_final(self):
        path = os.path.join(self.output_dir, "expert_data_final.pkl")
        with open(path, "wb") as f:
            pickle.dump({"success": self.success_samples, "fail": self.fail_contexts}, f)
        logger.info(f"Saved {len(self.success_samples)} samples to {path}")