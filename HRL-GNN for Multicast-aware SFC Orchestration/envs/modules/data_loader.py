import pickle
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器：管理请求序列和事件序列"""

    def __init__(self, config: dict):
        self.cfg = config
        self.requests = []
        self.events = []
        self.req_map = {}
        self.time_step = 0
        self.total_steps = 0

    def load_dataset(self, phase: str) -> bool:
        """根据阶段加载数据"""
        data_dir = self.cfg['path'].get('expert_data_dir', 'data/expert')
        # 根据 phase 自动推断文件名
        req_file = f"{phase}_requests.pkl"
        evt_file = f"{phase}_events.pkl"

        req_path = os.path.join(data_dir, req_file)
        evt_path = os.path.join(data_dir, evt_file)

        if not os.path.exists(req_path) or not os.path.exists(evt_path):
            logger.error(f"Data files not found: {req_path}")
            return False

        try:
            with open(req_path, 'rb') as f:
                self.requests = pickle.load(f)
            self.req_map = {r['id']: r for r in self.requests}

            with open(evt_path, 'rb') as f:
                raw_events = pickle.load(f)

            # 格式化事件
            self.events = []
            for evt in raw_events:
                self.events.append({
                    'arrive': np.array(evt.get('arrive', []), dtype=int).flatten(),
                    'leave': np.array(evt.get('leave', []), dtype=int).flatten()
                })

            self.total_steps = len(self.events)
            self.reset()
            logger.info(f"Loaded {phase} dataset: {len(self.requests)} requests, {self.total_steps} steps.")
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def reset(self):
        self.time_step = 0

    def get_current_arrivals(self) -> list:
        """获取当前时间步到达的请求"""
        if self.time_step >= self.total_steps:
            return []

        arrive_ids = self.events[self.time_step]['arrive']
        reqs = []
        for rid in arrive_ids:
            if rid in self.req_map:
                reqs.append(self.req_map[rid])
        return reqs

    def get_current_leaves(self) -> list:
        """获取当前时间步离开的请求ID"""
        if self.time_step >= self.total_steps:
            return []
        return self.events[self.time_step]['leave'].tolist()

    def advance_time(self):
        self.time_step += 1

    def is_done(self):
        return self.time_step >= self.total_steps