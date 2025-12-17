# ================================================================
# Phase 1 Expert Data Collector (FINAL)
# Continuous Simulation + Request-level Expert
# ================================================================

import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    """
    Phase 1: Expert Data Collection (Continuous Simulation)

    Success:
        - Expert builds multicast tree
        - Tree is fully deployable under current residual resources

    Failure:
        - Expert cannot find feasible deployment
        - Or resource exhausted during execution
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

    def _estimate_load(self) -> float:
        """
        估算当前网络负载（用于 Phase2 区分轻载 / 重载）
        返回值 ∈ [0, 1]
        """
        rm = self.env.resource_mgr

        # 带宽利用率
        bw_util = 1.0 - rm.B.mean() / max(1.0, rm.B_cap)

        # CPU 利用率
        cpu_util = 1.0 - rm.C.mean() / max(1.0, rm.C_cap)

        # 简单平均
        return 0.5 * bw_util + 0.5 * cpu_util

    # ------------------------------------------------------------
    # Main entry (CONTINUOUS MODE)
    # ------------------------------------------------------------
    def collect(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection (Continuous Mode)")

        # 🔥 Only reset ONCE: empty network, start simulation
        # 务必加上 options={'phase': 'phase1'}，否则它会去跑 Phase 3 的数据
        self.env.reset(options={'phase': 'phase1'})

        pbar = tqdm(range(self.max_requests), desc="Collecting", ncols=120)

        for idx in pbar:
            # 🔥 Advance time & get next arriving request
            req, state = self.env.reset_request()

            if req is None:
                logger.info("✓ No more requests in dataset, stop Phase 1")
                break

            self.stats["requests"] += 1
            req_id = req.get("id", "unknown")

            # ----------------------------------------------------
            # Expert solves under CURRENT residual resources
            # ----------------------------------------------------
            network_state = self.env.resource_mgr.get_network_state_dict(req)

            expert_result = self.expert.solve_request_for_expert(
                request=req,
                network_state=network_state
            )

            success = False

            if expert_result is not None:
                tree, traj = expert_result

                # Try to deploy on env (真实资源扣减)
                # 使用 tree 专用接口，更稳健
                ok = self.env.resource_mgr.apply_tree_deployment(req, tree)

                if ok:
                    success = True

                    # register service for future leave
                    self.env.event_handler.register_service(req_id, tree)

                    for (dest_idx, action, cost) in traj:
                        self.success_samples.append({
                            "request": req,
                            "dest_idx": dest_idx,
                            "action": action,
                            "cost": cost,
                            # 可选：记录当时的负载，用于 Phase2 分流
                            "load": self._estimate_load(),
                        })

            # ----------------------------------------------------
            # Failure handling (BLOCKING)
            # ----------------------------------------------------
            if not success:
                self.stats["fail"] += 1

                # record fail context for Phase2
                self.fail_contexts.append({
                    "request": req,
                    "network_state": network_state,
                    "reason": "resource_blocking"
                })
            else:
                self.stats["success"] += 1

            # ----------------------------------------------------
            # Logging & saving
            # ----------------------------------------------------
            if (idx + 1) % self.save_every == 0:
                self._save_partial(idx + 1)

            br = self.stats["fail"] / max(1, self.stats["requests"])
            pbar.set_postfix(
                succ=self.stats["success"],
                fail=self.stats["fail"],
                BR=f"{br:.2%}"
            )

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
