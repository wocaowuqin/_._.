import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    """
    Phase 1: Expert Data Collection (Continuous Simulation)

    Success:
        - Expert builds multicast tree
        - Tree is fully deployable under current residual resources

    Failure:
        - Expert cannot find feasible deployment
    """

    def __init__(
        self,
        env,
        expert_solver,
        output_dir: str,
        max_episodes: int = 2000,
        save_every: int = 500,
    ):
        self.env = env
        self.expert = expert_solver

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_requests = max_episodes
        self.save_every = save_every

        # buffers
        self.success_samples: List[Dict[str, Any]] = []
        self.fail_contexts: List[Dict[str, Any]] = []

        # stats
        self.stats = {
            "requests": 0,
            "success": 0,
            "fail": 0,
        }

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def _estimate_load(self) -> float:
        """
        Rough network load estimation ∈ [0, 1]
        """
        rm = self.env.resource_mgr

        bw_util = 1.0 - rm.B.mean() / max(1.0, rm.B_cap)
        cpu_util = 1.0 - rm.C.mean() / max(1.0, rm.C_cap)

        return 0.5 * bw_util + 0.5 * cpu_util

    # ------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------
    def collect(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection (GNN Mode)")

        # 1. 重置环境
        self.env.reset(options={'phase': 'phase1'})

        # 2. 获取数据源 (兼容 env 不同实现)
        if hasattr(self.env, 'data_loader'):
            events = self.env.data_loader.events
            requests = self.env.data_loader.requests
        else:
            # 兼容旧接口
            events = getattr(self.env, 'events', [])
            requests = getattr(self.env, 'requests', [])

        if not requests:
            logger.error("❌ No requests found! Please check env initialization.")
            return

        total_events = len(events)
        pbar = tqdm(total=total_events, desc="Collecting", ncols=120)

        for t, event in enumerate(events):
            pbar.update(1)

            # 兼容不同的键名 (arrive / arrive_event)
            arrive_list = event.get("arrive", event.get("arrive_event", []))
            leave_list = event.get("leave", event.get("leave_event", []))

            # 处理离开
            for leave_req_id in leave_list:
                try:
                    self.env.event_handler.unregister_service(leave_req_id)
                except Exception:
                    pass

            # 处理到达
            for req_id in arrive_list:
                # ID 校验
                if req_id <= 0 or req_id > len(requests):
                    continue

                req = requests[req_id - 1]
                self.stats["requests"] += 1

                # ========================================================
                # 📸 [核心修复]：捕获当前的图状态 (Graph State)
                # ========================================================
                # 1. 手动同步环境状态，以便 get_graph_state 能读到正确数据
                self.env.current_request = req
                # 初始化临时树状态（专家决策前，树只包含源节点）
                current_nodes = {req['source']}
                current_tree = {
                    'tree': np.zeros(self.env.resource_mgr.L),
                    'hvt': np.zeros((self.env.resource_mgr.n, 8)),
                    'links': []
                }

                # 2. 调用 ResourceManager 生成图特征
                # 注意：这里直接调用 resource_mgr 的方法，绕过 env.get_state 的模式判断
                x, edge_index, edge_attr, req_vec = self.env.resource_mgr.get_graph_state(
                    current_request=req,
                    nodes_on_tree=current_nodes,
                    current_tree=current_tree,
                    served_dest_count=0,
                    sharing_strategy=0,
                    nb_high_goals=10  # 假设高层目标数为10
                )

                # 3. 封装为 PyG Data 对象 (只保存 Tensor，节省空间)
                # 必须 .cpu() 确保不占用 GPU 显存
                state_to_save = Data(
                    x=x.cpu(),
                    edge_index=edge_index.cpu(),
                    edge_attr=edge_attr.cpu(),
                    req_vec=req_vec.cpu()
                )
                # ========================================================

                # 获取网络状态字典供专家使用
                network_state = self.env.resource_mgr.get_network_state_dict(req)

                # 调用专家求解
                expert_result = self.expert.solve_request_for_expert(req, network_state)

                success = False
                if expert_result is not None:
                    tree, traj = expert_result
                    if tree is not None:
                        # 尝试部署
                        if self.env.resource_mgr.apply_tree_deployment(req, tree):
                            success = True
                            self.stats["success"] += 1
                            self.env.event_handler.register_service(req_id, tree)

                            # ✅ [修复] 保存样本时，带上 "state"
                            for (dest_idx, action, cost) in traj:
                                self.success_samples.append({
                                    "state": state_to_save,  # <--- 必须有这个！
                                    "request": req,
                                    "high_action": dest_idx,  # 统一命名
                                    "dest_idx": dest_idx,
                                    "action": action,
                                    "cost": cost,
                                    "load": self._estimate_load(),
                                })

                if not success:
                    self.stats["fail"] += 1
                    self.fail_contexts.append({
                        "request": req,
                        "network_state": network_state,
                        "reason": "expert_failed"
                    })

                br = self.stats["fail"] / max(1, self.stats["requests"])
                pbar.set_postfix(succ=self.stats["success"], fail=self.stats["fail"], BR=f"{br:.2%}")

            # 提前结束
            if self.stats["requests"] >= self.max_requests:
                logger.info("✓ Reach max request limit")
                break

            # 定期保存
            if self.stats["requests"] > 0 and self.stats["requests"] % self.save_every == 0:
                self._save_final()

        pbar.close()
        self._save_final()
        logger.info(f"✓ Phase 1 Done. Stats: {self.stats}")
        return self.stats
    # ------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------
    def _save_partial(self, idx: int):
        path = os.path.join(self.output_dir, f"expert_data_part_{idx}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "success": self.success_samples,
                "fail": self.fail_contexts
            }, f)
        logger.info(f"[Expert] Saved partial data to {path}")

    def _save_final(self):
        path = os.path.join(self.output_dir, "expert_data_final.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "success": self.success_samples,
                "fail": self.fail_contexts
            }, f)
        logger.info(f"[Expert] Saved final data to {path}")
