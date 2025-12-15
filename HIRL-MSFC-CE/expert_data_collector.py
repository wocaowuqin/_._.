#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expert_data_collector.py (Fixed Version)
Phase1 专家数据收集器 — 修复 __init__ 参数错误

关键修复：
1. ✅ def __init__(..., config: Optional[dict] = None) <--- 修复报错的关键
2. ✅ 强制资源重置 (防止 200ep 后失败)
3. ✅ 集成置信度计算
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import logging
import time
import random
import json
import numpy as np
import traceback

# 尝试导入置信度计算器
try:
    from expert_confidence_calculator import ExpertConfidenceCalculator
except ImportError:
    # 哑类防止报错
    class ExpertConfidenceCalculator:
        def __init__(self, config=None): pass

        def compute_confidence(self, **kwargs): return 0.5

        def update_feedback(self, **kwargs): pass

        def print_report(self): pass

        def health_check(self): return {'status': 'dummy'}

        def save_state(self, path): pass

logger = logging.getLogger(__name__)


class EnhancedStats:
    """统计追踪辅助类"""

    def __init__(self):
        self.stats = {
            "episodes_run": 0,
            "episodes_success": 0,
            "episodes_failed": 0,
            "transitions_collected": 0,
            "files_saved": 0
        }
        self.start_time = time.time()

    def update(self, meta: dict, num_transitions: int):
        self.stats["episodes_run"] += 1
        if meta.get("complete", False):
            self.stats["episodes_success"] += 1
        else:
            self.stats["episodes_failed"] += 1
        self.stats["transitions_collected"] += num_transitions

    def get_summary(self) -> Dict[str, Any]:
        s = self.stats.copy()
        if s["episodes_run"] > 0:
            s["success_rate"] = s["episodes_success"] / s["episodes_run"]
        return s


class ExpertDataCollector:
    # [关键修复] 必须包含 config 参数
    def __init__(self, env, output_dir: str = "output/expert", config: Optional[dict] = None):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        default_cfg = {
            "episodes": 1000,
            "save_every": 200,
            "flush_on_save": True,
            "expert_randomness": 0.1,
            "max_retries": 3,
            "augment_data": False,
            "confidence_config": {
                "strategy": "hybrid",
                "min_conf": 0.25,
                "max_conf": 0.95
            }
        }
        # 合并配置
        self.cfg = {**default_cfg, **(config or {})}

        self.stats = EnhancedStats()
        self.buffer: List[Dict[str, Any]] = []
        self.saved_files: List[str] = []

        logger.info("Initializing ExpertConfidenceCalculator...")
        self.conf_calculator = ExpertConfidenceCalculator(
            config=self.cfg.get("confidence_config", {})
        )

    def run(self) -> List[str]:
        episodes = int(self.cfg.get("episodes", 1000))
        save_every = int(self.cfg.get("save_every", 200))
        start_time = time.time()

        if not hasattr(self, 'saved_files'):
            self.saved_files = []

        logger.info(f"Starting Phase1 Collection. Target: {episodes}")

        for i in range(1, episodes + 1):
            try:
                # [Fix] 强制资源重置
                self._force_reset_resources()

                traj, meta = self._collect_with_retry(i)
            except Exception as e:
                logger.error(f"Ep {i} error: {e}")
                traj, meta = None, {}

            self.stats.update(meta, len(traj) if traj else 0)

            if traj:
                if self.cfg.get("augment_data", False):
                    traj = self._augment_data(traj)
                self.buffer.extend(traj)

            if i % save_every == 0 or i == episodes:
                self._print_progress(i, episodes, start_time)
                if self.buffer:
                    self._flush_buffer(episode_idx=i)

        if self.buffer:
            self._flush_buffer(episode_idx=episodes)

        # 保存置信度状态
        try:
            self.conf_calculator.save_state(str(self.output_dir / "confidence_state.pkl"))
        except:
            pass

        self._log_stats()
        return self.saved_files

    def _force_reset_resources(self):
        """强制重置网络资源，防止累积效应"""
        try:
            if hasattr(self.env, 'reset_network'):
                self.env.reset_network()
            elif hasattr(self.env, 'capacities'):
                # 手动重置
                cap = self.env.capacities
                self.env.C = np.full(self.env.n, cap.get('cpu', 100.0), dtype=np.float32)
                self.env.M = np.full(self.env.n, cap.get('memory', 100.0), dtype=np.float32)
                self.env.B = np.full(self.env.L, cap.get('bandwidth', 100.0), dtype=np.float32)
                if hasattr(self.env, 'hvt_all'):
                    self.env.hvt_all = np.zeros((self.env.n, 8), dtype=np.float32)
        except Exception as e:
            logger.warning(f"Reset failed: {e}")

    def _collect_with_retry(self, episode_idx: int) -> Tuple[Optional[List], dict]:
        max_retries = self.cfg.get("max_retries", 1)
        for attempt in range(max_retries):
            try:
                return self._collect_single_episode()
            except Exception as e:
                if attempt == max_retries - 1:
                    return None, {"complete": False}
        return None, {"complete": False}

    def _collect_single_episode(self) -> Tuple[Optional[List[Dict[str, Any]]], dict]:
        req, state = self.env.reset_request()
        if req is None: return None, {}

        request_id = req.get("id", random.randint(0, 9999))

        try:
            # 准备 Expert 输入
            import numpy as np

            import numpy as np

            def safe_copy(obj):
                """安全复制：支持 tuple/list/ndarray"""
                if isinstance(obj, np.ndarray):
                    return obj.copy()
                else:
                    return np.array(obj)

            network_state = {
                'cpu': safe_copy(self.env.C),
                'mem': safe_copy(self.env.M),
                'bw': safe_copy(self.env.B),
                'hvt': safe_copy(self.env.hvt_all),
                'bw_ref_count': getattr(self.env, 'bw_ref_count', np.zeros(self.env.expert.link_num))
            }

            expert_tree, expert_traj = self.env.expert.solve_request_for_expert(req, network_state)

            if expert_tree is not None and expert_traj:
                episode_traj = []
                req_info = {'dest': req.get('dest', [])}
                complete = len(expert_tree.get('added_dest_indices', [])) == len(req['dest'])

                # 计算置信度
                cost_sum = sum(t[2] for t in expert_traj) if expert_traj else 0
                conf = self.conf_calculator.compute_confidence(
                    cost=cost_sum, request_info=req_info, expert_tree=expert_tree, episode_id=str(request_id)
                )
                self.conf_calculator.update_feedback(cost_sum, conf, 1.0 if complete else 0.5)

                for traj_item in expert_traj:
                    dest_idx, action_tuple, cost = traj_item
                    path_idx, k_idx, placement = action_tuple

                    # 🔥 修复：安全复制 state
                    import copy as copy_module
                    if isinstance(state, np.ndarray):
                        state_copy = state.copy()
                    else:
                        state_copy = copy_module.deepcopy(state)

                    transition = {
                        "state": state_copy,
                        "goal": dest_idx,
                        "action": path_idx,
                        "next_state": state_copy,
                        "reward": -cost,
                        "confidence": conf,
                        "request_id": request_id,
                        "cost": cost
                    }
                    episode_traj.append(transition)

                return episode_traj, {"complete": complete, "expert_used": True}

            else:
                return self._collect_with_heuristic(req, state, request_id)

        except Exception as e:
            logger.error(f"Expert solve error: {e}")
            return self._collect_with_heuristic(req, state, request_id)

    def _collect_with_heuristic(self, req, state, request_id):
        # 简单启发式 Fallback
        unadded = getattr(self.env, "unadded_dest_indices", set())
        if not unadded: return None, {}

        episode_traj = []
        dummy_cost = 200.0
        conf = self.conf_calculator.compute_confidence(dummy_cost, episode_id=str(request_id))
        self.conf_calculator.update_feedback(dummy_cost, conf, 0.0)

        for dest_idx in list(unadded):
            valid_actions = self._get_valid_actions(None)
            if not valid_actions: break
            action = random.choice(valid_actions)

            transition = {
                "state": state, "goal": dest_idx, "action": action,
                "next_state": state, "reward": -10.0, "confidence": conf,
                "request_id": request_id, "cost": dummy_cost
            }
            episode_traj.append(transition)

        return episode_traj, {"complete": False, "heuristic_used": True}

    def _get_valid_actions(self, node):
        if hasattr(self.env, "get_valid_low_level_actions"):
            return list(self.env.get_valid_low_level_actions())
        return list(range(getattr(self.env, 'n', 28)))

    def _augment_data(self, buffer):
        return buffer  # 简化

    def _print_progress(self, current, total, start_time):
        elapsed = time.time() - start_time
        logger.info(f"Progress: {current}/{total} | Time: {elapsed:.1f}s")

    def _flush_buffer(self, episode_idx):
        if not self.buffer: return
        filename = f"expert_data_part_{self.stats.stats['files_saved']}_{episode_idx}.pkl"
        path = self.output_dir / filename
        payload = {"expert_data": self.buffer, "stats": self.stats.get_summary()}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        self.saved_files.append(str(path))
        self.stats.stats["files_saved"] += 1
        if self.cfg.get("flush_on_save", True): self.buffer = []

    def _log_stats(self):
        logger.info(f"Collection Stats: {self.stats.get_summary()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ExpertDataCollector module loaded.")