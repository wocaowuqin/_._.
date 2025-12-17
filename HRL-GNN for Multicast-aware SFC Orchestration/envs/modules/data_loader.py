import pickle
import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器：最终修复版 (支持 arrive_event/leave_event 键名)"""

    def __init__(self, config: dict):
        self.cfg = config
        self.requests = []
        self.events = []
        self.req_map = {}
        self.time_step = 0
        self.total_steps = 0

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        # --- 路径查找逻辑 (保持不变) ---
        if events_file is not None:
            req_filename = phase_or_req_file
            possible_dirs = [
                self.cfg['path'].get('expert_data_dir', 'data/expert'),
                'generate_requests_depend_on_poisson/data_output',
                'data_output',
                'data/expert',
                self.cfg['path'].get('input_dir', 'data/input_dir'),
                '.'
            ]
            req_path, evt_path = None, None
            for search_dir in possible_dirs:
                if not search_dir: continue
                tr = os.path.join(search_dir, req_filename)
                te = os.path.join(search_dir, events_file)
                if os.path.exists(tr) and os.path.exists(te):
                    req_path, evt_path = tr, te
                    logger.info(f"Found data files in: {search_dir}")
                    break
            if not req_path:
                logger.error(f"Data files not found: {req_filename}, {events_file}")
                return False
            return self._load_from_paths(req_path, evt_path)
        else:
            phase = phase_or_req_file
            data_dir = self.cfg['path'].get('input_dir', 'data/input_dir')
            req_path = os.path.join(data_dir, f"{phase}_requests.pkl")
            evt_path = os.path.join(data_dir, f"{phase}_events.pkl")
            if not os.path.exists(req_path) or not os.path.exists(evt_path):
                alt_dir = self.cfg['path'].get('expert_data_dir', 'data/expert')
                req_path = os.path.join(alt_dir, f"{phase}_requests.pkl")
                evt_path = os.path.join(alt_dir, f"{phase}_events.pkl")
                if not os.path.exists(req_path) or not os.path.exists(evt_path):
                    logger.error(f"Data files not found for phase '{phase}'")
                    return False
            return self._load_from_paths(req_path, evt_path)

    def _load_from_paths(self, req_path, evt_path):
        try:
            # 1. 加载请求
            with open(req_path, 'rb') as f:
                self.requests = pickle.load(f)

            self.req_map = {}
            for r in self.requests:
                rid = int(r['id'])
                r['id'] = rid
                self.req_map[rid] = r

            # 2. 加载事件
            with open(evt_path, 'rb') as f:
                raw_events = pickle.load(f)

            # 3. 🔥 [关键修复] 适配 arrive_event / leave_event
            self.events = []
            all_event_ids = set()

            for i, evt in enumerate(raw_events):
                arr, lv = [], []

                # 情况 A: 字典格式
                if isinstance(evt, dict):
                    # 🔥 优先检查 arrive_event (您的数据格式)
                    arr = evt.get('arrive_event', evt.get('arrive', evt.get('arrived', [])))
                    lv = evt.get('leave_event', evt.get('leave', evt.get('left', [])))

                # 情况 B: 列表/元组格式
                elif isinstance(evt, (list, tuple, np.ndarray)):
                    if len(evt) >= 1: arr = evt[0]
                    if len(evt) >= 2: lv = evt[1]

                # 转换为 int 列表
                arr = np.array(arr, dtype=int).flatten().tolist()
                lv = np.array(lv, dtype=int).flatten().tolist()

                self.events.append({'arrive': arr, 'leave': lv})
                all_event_ids.update(arr)

            self.total_steps = len(self.events)
            self.reset()

            # 4. ID 对齐检查与修正
            if len(all_event_ids) > 0:
                req_ids = set(self.req_map.keys())
                overlap = req_ids.intersection(all_event_ids)

                if len(overlap) == 0:
                    logger.warning("⚠️ Request/Event ID 不匹配，正在尝试修复...")

                    # 尝试偏移 -1
                    shifted_down = {x - 1 for x in all_event_ids}
                    if len(req_ids.intersection(shifted_down)) > 0:
                        logger.info("🔧 修复: Event ID - 1 (1-based -> 0-based)")
                        for e in self.events:
                            e['arrive'] = [x - 1 for x in e['arrive']]
                            e['leave'] = [x - 1 for x in e['leave']]
                    # 尝试偏移 +1
                    else:
                        shifted_up = {x + 1 for x in all_event_ids}
                        if len(req_ids.intersection(shifted_up)) > 0:
                            logger.info("🔧 修复: Event ID + 1")
                            for e in self.events:
                                e['arrive'] = [x + 1 for x in e['arrive']]
                                e['leave'] = [x + 1 for x in e['leave']]
                        else:
                            # 您的请求ID是 1, 20... 如果事件里是 0, 19... 则需要+1
                            logger.warning("❌ 无法自动对齐 ID，将按原样尝试...")
            else:
                logger.warning(f"⚠️ 加载了 {len(self.events)} 个时间步，但似乎所有 arrive_event 都是空的？")

            logger.info(f"Loaded dataset: {len(self.requests)} requests, {self.total_steps} steps")
            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reset(self):
        self.time_step = 0

    def get_current_arrivals(self) -> list:
        if self.time_step >= self.total_steps: return []
        arrive_ids = self.events[self.time_step]['arrive']
        return [self.req_map[rid] for rid in arrive_ids if rid in self.req_map]

    def get_current_leaves(self) -> list:
        if self.time_step >= self.total_steps: return []
        return self.events[self.time_step]['leave']

    def advance_time(self):
        self.time_step += 1

    def is_done(self):
        return self.time_step >= self.total_steps