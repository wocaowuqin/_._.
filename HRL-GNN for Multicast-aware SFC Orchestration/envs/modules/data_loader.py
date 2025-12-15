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
        """
        根据阶段或文件名加载数据集（支持两种调用方式）

        调用方式1: load_dataset("phase3")
                  自动推断文件名为 phase3_requests.pkl 和 phase3_events.pkl

        调用方式2: load_dataset("phase3_requests.pkl", "phase3_events.pkl")
                  直接指定文件名（旧环境兼容）

        Args:
            phase_or_req_file: 阶段名（如 "phase3"）或请求文件名
            events_file: 事件文件名（可选，提供时使用文件名方式）

        Returns:
            bool: 加载是否成功
        """
        # 方式2：文件名方式（旧环境兼容）
        if events_file is not None:
            # 尝试多个可能的数据目录
            possible_dirs = [
                self.cfg['path'].get('expert_data_dir', 'data/expert'),
                'generate_requests_depend_on_poisson/data_output',
                'data_output',
                'data/expert',
                '.'
            ]

            req_path = None
            evt_path = None

            # 查找文件
            for data_dir in possible_dirs:
                test_req = os.path.join(data_dir, phase_or_req_file)
                test_evt = os.path.join(data_dir, events_file)

                if os.path.exists(test_req) and os.path.exists(test_evt):
                    req_path = test_req
                    evt_path = test_evt
                    logger.info(f"Found data files in: {data_dir}")
                    break

            if req_path is None or evt_path is None:
                logger.error(f"Data files not found: {phase_or_req_file}, {events_file}")
                logger.error(f"Searched in: {possible_dirs}")
                return False

            # 加载文件
            try:
                # 加载请求
                with open(req_path, 'rb') as f:
                    self.requests = pickle.load(f)
                self.req_map = {r['id']: r for r in self.requests}

                # 加载事件
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

                logger.info(f"Loaded files: {len(self.requests)} requests, {self.total_steps} steps")
                return True

            except Exception as e:
                logger.error(f"Load failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        # 方式1：阶段名方式（新环境）
        else:
            data_dir = self.cfg['path'].get('expert_data_dir', 'data/expert')

            # 根据 phase 自动推断文件名
            req_file = f"phase_requests.pkl"
            evt_file = f"phase_events.pkl"

            req_path = os.path.join(data_dir, req_file)
            evt_path = os.path.join(data_dir, evt_file)

            if not os.path.exists(req_path) or not os.path.exists(evt_path):
                logger.error(f"Data files not found: {req_path}")
                return False

            try:
                # 加载请求
                with open(req_path, 'rb') as f:
                    self.requests = pickle.load(f)
                self.req_map = {r['id']: r for r in self.requests}

                # 加载事件
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

                logger.info(
                    f"Loaded file dataset: {len(self.requests)} requests, {self.total_steps} steps")
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