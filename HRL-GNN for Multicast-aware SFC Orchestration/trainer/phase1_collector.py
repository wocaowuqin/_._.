# trainer/phase1_collector.py
import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from torch_geometric.data import Data
import torch

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

    def _sanitize_request(self, req):
        """🔥 [关键修复] 将 Request 对象转换为普通字典，防止 Pickle 报错"""
        if isinstance(req, dict):
            return req.copy()
        if hasattr(req, '__dict__'):
            return req.__dict__.copy()
        # 如果是其他类型，尝试强制转换
        try:
            return dict(req)
        except:
            # 最后的兜底：手动提取关键字段
            return {
                'id': getattr(req, 'id', -1),
                'source': getattr(req, 'source', 0),
                'dest': getattr(req, 'dest', []),
                'vnf': getattr(req, 'vnf', []),
                'bandwidth': getattr(req, 'bandwidth', 1.0),
                'ttl': getattr(req, 'ttl', 100)
            }

    def collect(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection (Pickle Safe Version)")
        # 强制重置
        self.env.reset()

        # 获取数据集引用
        if hasattr(self.env, 'data_loader'):
            events = self.env.data_loader.events
            requests = self.env.data_loader.requests
        else:
            logger.error("❌ Data loader not found!")
            return

        if not requests:
            logger.error("❌ No requests found!")
            return

        pbar = tqdm(total=min(len(events), self.max_requests * 2), desc="Collecting HRL Data", ncols=120)

        for t, event in enumerate(events):
            # 处理离开事件 (释放资源)
            leave_list = event.get("leave", event.get("leave_event", []))
            for leave_req_id in leave_list:
                try:
                    self.env.event_handler.unregister_service(leave_req_id)
                except:
                    pass

            # 处理到达事件
            arrive_list = event.get("arrive", event.get("arrive_event", []))
            for req_id in arrive_list:
                if req_id <= 0 or req_id > len(requests): continue
                req = requests[req_id - 1]
                self.stats["requests"] += 1
                pbar.update(1)

                # 🔥 1. 专家求解
                network_state = self.env.resource_mgr.get_network_state_dict(req)
                expert_result = self.expert.solve_request_for_expert(req, network_state)

                success = False
                if expert_result is not None:
                    tree_data, expert_traj = expert_result

                    if tree_data is not None:
                        # 转换树结构
                        dict_tree = {}
                        paths_map = tree_data.get('paths_map', {})

                        # 准备 HRL 轨迹
                        # ... (此处逻辑保持不变)

                        # 构造部署计划
                        deployment_plan = {
                            'hvt': tree_data['hvt'],
                            'tree': dict_tree
                        }

                        # 补充边信息（用于 Env 验证）
                        for dest, path_nodes in paths_map.items():
                            path_0 = [n - 1 if n > 0 else 0 for n in path_nodes]
                            for i in range(len(path_0) - 1):
                                u, v = path_0[i], path_0[i + 1]
                                dict_tree[(u, v)] = 1.0
                                dict_tree[(v, u)] = 1.0

                        # 2. 在 Env 中尝试部署
                        if self.env.resource_mgr.apply_tree_deployment(deployment_plan, req):
                            success = True
                            self.stats["success"] += 1

                            # 注册服务
                            self.env.event_handler.services[req_id] = {
                                'req': req, 'tree': deployment_plan, 'hvt': tree_data['hvt']
                            }

                            # 3. 构造样本
                            self.env.current_request = req
                            dummy_tree = {'tree': {}, 'hvt': np.zeros((self.env.n, 8))}
                            x, edge_index, edge_attr, req_vec = self.env.resource_mgr.get_graph_state(
                                current_request=req, nodes_on_tree={req['source']},
                                current_tree=dummy_tree, served_dest_count=0, sharing_strategy=0, nb_high_goals=10
                            )
                            state_to_save = Data(
                                x=x.cpu(),
                                edge_index=edge_index.cpu(),
                                edge_attr=edge_attr.cpu(),
                                req_vec=req_vec.cpu()
                            )

                            # 提取路径用于保存
                            paths_to_save = []
                            for dest, path_nodes in paths_map.items():
                                path_0 = [n - 1 if n > 0 else 0 for n in path_nodes]
                                paths_to_save.append(path_0)

                            # 🔥 [关键修复] 将 req 清洗为普通字典
                            clean_req = self._sanitize_request(req)

                            for path in paths_to_save:
                                self.success_samples.append({
                                    "state": state_to_save,
                                    "request": clean_req,  # <--- 这里存的是清洗后的字典
                                    "action": {"path": path},
                                    "cost": 0.0,
                                    "load": self._estimate_load(),
                                    "hrl_info": {
                                        "subgoal": path[-1],
                                        "full_path": path
                                    }
                                })

                if not success:
                    self.stats["fail"] += 1

                br = self.stats["fail"] / max(1, self.stats["requests"])
                pbar.set_postfix(succ=self.stats["success"], fail=self.stats["fail"], BR=f"{br:.1%}")

            if self.stats["success"] >= self.max_requests:
                break
            if self.stats["requests"] > 0 and self.stats["requests"] % self.save_every == 0:
                self._save_final()

        pbar.close()
        self._save_final()
        return self.stats

    def _save_final(self):
        path = os.path.join(self.output_dir, "expert_data_final.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump({"success": self.success_samples}, f)
            logger.info(f"✅ Saved {len(self.success_samples)} expert samples to {path}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            import traceback
            traceback.print_exc()