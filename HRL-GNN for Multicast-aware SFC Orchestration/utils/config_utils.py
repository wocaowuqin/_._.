import yaml
import os
import argparse
from typing import Dict, Any


def get_project_root():
    """
    🔥 核心修复：自动获取项目根目录
    逻辑：当前脚本在 utils/ 下，其父目录的父目录就是项目根目录（如果 utils 是根目录下的直接子文件夹）
    或者：当前脚本在 utils/ 下，其父目录就是项目根目录
    """
    # 获取当前脚本的绝对路径: .../your_project/utils/config_utils.py
    current_file_path = os.path.abspath(__file__)
    # 获取 utils 目录路径: .../your_project/utils
    utils_dir = os.path.dirname(current_file_path)
    # 获取项目根目录: .../your_project
    project_root = os.path.dirname(utils_dir)
    return project_root


def load_yaml(file_path: str) -> Dict[str, Any]:
    """安全加载单个 YAML 文件"""
    if not os.path.exists(file_path):
        # 打印出绝对路径，方便排查
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file {file_path}: {e}")


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """递归深度更新字典"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(phase: str = 'phase3', config_dir: str = None) -> Dict[str, Any]:
    """
    加载并合并配置
    """
    # 🔥 自动定位 configs 目录的绝对路径
    if config_dir is None:
        root = get_project_root()
        config_dir = os.path.join(root, 'configs')

    # 1. 加载基础配置
    base_path = os.path.join(config_dir, 'base.yaml')
    config = load_yaml(base_path)

    # 2. 加载模型配置
    model_path = os.path.join(config_dir, 'model.yaml')
    model_cfg = load_yaml(model_path)
    deep_update(config, model_cfg)

    # 3. 加载特定阶段配置
    if phase:
        phase_path = os.path.join(config_dir, f'{phase}.yaml')
        phase_cfg = load_yaml(phase_path)
        deep_update(config, phase_cfg)
        config['current_phase'] = phase

    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='phase3')
    return parser.parse_args()


if __name__ == "__main__":
    # 测试代码
    try:
        # 打印出推导的根目录，帮你确认路径对不对
        print(f"📍 项目根目录定位为: {get_project_root()}")
        print(f"📍 配置目录定位为: {os.path.join(get_project_root(), 'configs')}")

        cfg = load_config('phase3')
        print("✅ 配置加载成功！")

        print("\n[验证内容]")
        print(f"1. Node Num (base): {cfg['env'].get('node_num', 'N/A')}")
        print(f"2. LR (phase3): {cfg['rl'].get('lr', 'N/A')}")
        print(f"3. Hidden Dim (model): {cfg['gnn'].get('hidden_dim', 'N/A')}")

    except Exception as e:
        print(f"❌ 错误: {e}")