from typing import List, Dict, Optional


class PathManager:
    """路径管理器：缓存和索引路径，减少重复计算"""

    def __init__(self, max_paths=10):
        self.max_paths = max_paths
        self.paths: List[List[int]] = []
        self.path_to_idx: Dict[tuple, int] = {}

    def add_path(self, path: List[int]) -> int:
        path_tuple = tuple(path)
        if path_tuple in self.path_to_idx:
            return self.path_to_idx[path_tuple]

        # 简单策略：如果满了就不存了，或者FIFO（此处保持简化）
        if len(self.paths) < self.max_paths:
            idx = len(self.paths)
            self.paths.append(path)
            self.path_to_idx[path_tuple] = idx
            return idx
        return 0  # 默认/失败索引

    def get_path(self, idx: int) -> Optional[List[int]]:
        if 0 <= idx < len(self.paths):
            return self.paths[idx]
        return None

    def reset(self):
        self.paths.clear()
        self.path_to_idx.clear()

    def __len__(self):
        return len(self.paths)