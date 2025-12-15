#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段 HIRL-SFC 完整训练系统
Phase 1: 专家轨迹采集 (使用 expert_data_collector.py)
Phase 2: 监督模仿学习 (使用 Agent 的 supervised_update)
Phase 3: 强化学习微调 (使用 curriculum_scheduler.py)

整合所有现有模块，开箱即用
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

# 导入项目模块
import hyperparameters as H
from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN
from hirl_sfc_agent_gnn import Agent_SFC_GNN
from expert_data_collector import ExpertDataCollector
from curriculum_scheduler import CurriculumScheduler
from multicast_aware_gat import MulticastAwareGAT
from multicast_gat_wrapper_vectorized import MulticastGATWrapperVectorized
from pathlib import Path
# 导入训练系统
from train_three_phase_complete import (
    HIRLThreePhaseTrainer,
    Phase1ExpertCollector,
    Phase2ILTrainer,
    Phase3RLTrainer
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(H.OUTPUT_DIR / 'three_phase_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
#   Phase 1 适配器 - 使用你的 ExpertDataCollector
# =====================================================================

# 确保导入正确
from expert_data_collector import ExpertDataCollector


class Phase1AdaptedCollector:
    """
    适配器：连接 main.py 旧接口与新版 ExpertDataCollector
    """

    def __init__(self, env, output_dir, config):
        self.env = env
        self.output_dir = output_dir

        # [关键修复 1] 必须显式保存 config
        self.config = config

        # 实例化真正的新版收集器
        # 注意：这里我们不需要再手动传 config 中的参数了，run() 会自己处理
        self.real_collector = ExpertDataCollector(
            env=env,
            output_dir=output_dir,
            config=config
        )

    # main.py 中 Phase1AdaptedCollector 的完整修复版本
    # 替换第 77-130 行

    def run(self):
        """运行数据采集并返回转换后的轨迹数据"""
        logger.info("=" * 70)
        logger.info(f"Phase 1: 专家轨迹采集 (目标: {self.config.get('episodes', 1000)} episodes)")
        logger.info("=" * 70)

        # 1. 调用新版收集器
        saved_files = self.real_collector.run()

        # 2. 数据回读与格式转换
        trajectory = []
        all_confidences = []

        logger.info(f"正在从 {len(saved_files)} 个文件中加载数据...")

        from pathlib import Path
        import numpy as np
        output_dir = Path(self.output_dir)

        # ✅ 修复：查找 part 文件而不是 ep 文件
        part_files = list(output_dir.glob("expert_data_part_*.pkl"))

        if not part_files:
            logger.error("没有找到 expert_data_part_*.pkl 文件")
            raise ValueError("Phase 1 failed: No expert data collected!")

        logger.info(f"找到 {len(part_files)} 个 part 文件")

        # ✅ 加载所有 part 文件并合并数据
        for part_file in sorted(part_files):
            logger.info(f"加载: {part_file.name}")

            try:
                with open(part_file, 'rb') as f:
                    data_pkg = pickle.load(f)

                    # ✅ 使用 'expert_data' 键（这是你的 collector 保存的键）
                    raw_data = data_pkg.get("expert_data", [])

                    logger.info(f"  从 {part_file.name} 加载了 {len(raw_data)} 条数据")

                    for item in raw_data:
                        # 转换为 Tuple
                        traj = (
                            item['state'],
                            item['goal'],
                            item['action'],
                            item['reward'],
                            item['next_state']
                        )
                        trajectory.append(traj)

                        # ✅ 收集置信度
                        conf = item.get('confidence', 0.0)  # ← 注意是 'confidence' 不是 'expert_confidence'
                        all_confidences.append(conf)

            except Exception as e:
                logger.error(f"加载 {part_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not trajectory:
            raise ValueError("Phase 1 failed: No expert data collected!")

        # ✅ 计算真实的置信度统计
        if all_confidences:
            avg_conf = np.mean(all_confidences)
            min_conf = min(all_confidences)
            max_conf = max(all_confidences)
            zero_count = sum(1 for c in all_confidences if c == 0.0)

            logger.info(f"✓ Phase 1 完成: 收集了 {len(trajectory)} 条样本")
            logger.info(f"  - 平均置信度: {avg_conf:.3f}")
            logger.info(f"  - 置信度范围: {min_conf:.3f} - {max_conf:.3f}")
            logger.info(
                f"  - 零值样本: {zero_count}/{len(all_confidences)} ({zero_count / len(all_confidences) * 100:.1f}%)")
        else:
            logger.warning(f"✓ Phase 1 完成: 收集了 {len(trajectory)} 条样本")
            logger.warning(f"  ⚠️ 警告：没有置信度数据")

        return trajectory

    @property
    def stats(self):
        return self.real_collector.stats
    @property
    def stats(self):
        return self.real_collector.stats
# =====================================================================
#   Phase 2 适配器 - 使用 Agent 的监督学习功能
# =====================================================================

class Phase2AdaptedTrainer:
    """
    适配你的 Agent 的监督学习功能到三阶段系统
    """

    def __init__(self, agent, expert_data: List[Tuple],
                 output_dir: str, config: Optional[Dict] = None):
        self.agent = agent
        self.data = expert_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        default_config = {
            "epochs": 10,
            "batch_size": 32,
            "save_every_epoch": 2,
            "validation_split": 0.1
        }
        self.cfg = {**default_config, **(config or {})}

        # 划分训练集和验证集
        self._split_data()

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def _split_data(self):
        """划分训练集和验证集"""
        n_total = len(self.data)
        n_val = int(n_total * self.cfg["validation_split"])

        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.train_data = [self.data[i] for i in train_indices]
        self.val_data = [self.data[i] for i in val_indices]

        logger.info(f"数据划分 - 训练: {len(self.train_data)}, 验证: {len(self.val_data)}")

    def run(self) -> bool:
        """运行监督学习训练"""
        logger.info("=" * 70)
        logger.info("Phase 2: 监督模仿学习 (使用 Agent.supervised_update)")
        logger.info("=" * 70)

        # 切换到模仿学习模式
        self.agent.switch_to_imitation_mode()

        best_val_accuracy = 0.0

        for epoch in range(1, self.cfg["epochs"] + 1):
            # 训练
            train_loss, train_acc = self._train_epoch()

            # 验证
            val_loss, val_acc = self._validate()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            logger.info(
                f"Epoch {epoch}/{self.cfg['epochs']} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )

            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.agent.save(str(self.output_dir / "il_best.pth"))
                logger.info(f"✓ 新最佳模型 (验证准确率: {val_acc:.2f}%)")

            # 定期保存
            if epoch % self.cfg["save_every_epoch"] == 0:
                self.agent.save(str(self.output_dir / f"il_epoch{epoch}.pth"))

        # 保存最终模型
        self.agent.save(str(self.output_dir / "il_final.pth"))

        # 保存历史
        with open(self.output_dir / "training_history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        logger.info("=" * 70)
        logger.info(f"Phase 2 完成 | 最佳验证准确率: {best_val_accuracy:.2f}%")
        logger.info("=" * 70)

        return True

    def _train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch"""
        np.random.shuffle(self.train_data)

        batch_size = self.cfg["batch_size"]
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for i in range(0, len(self.train_data), batch_size):
            batch_transitions = self.train_data[i:i + batch_size]

            # 转换为 Agent 需要的格式
            batch_data = []
            for state, goal, action, reward, next_state in batch_transitions:
                # 获取有效动作
                valid_actions = self._get_valid_actions(state, goal)

                batch_data.append({
                    'state': state,
                    'goal': goal,
                    'action': action,
                    'valid_actions': valid_actions
                })

            # 调用 Agent 的 supervised_update
            loss, accuracy = self.agent.supervised_update(batch_data)

            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_accuracy = total_accuracy / max(1, num_batches)

        return avg_loss, avg_accuracy

    def _validate(self) -> Tuple[float, float]:
        """验证"""
        # 使用 Agent 的 evaluate_imitation
        eval_data = []
        for state, goal, action, reward, next_state in self.val_data:
            valid_actions = self._get_valid_actions(state, goal)
            eval_data.append({
                'state': state,
                'goal': goal,
                'action': action,
                'valid_actions': valid_actions
            })

        results = self.agent.evaluate_imitation(
            eval_data,
            num_samples=min(500, len(eval_data))
        )

        # 估算损失（简化）
        val_loss = 2.0 * (1.0 - results['accuracy'] / 100.0)
        val_accuracy = results['accuracy']

        return val_loss, val_accuracy

    def _get_valid_actions(self, state, goal):
        """获取有效动作列表"""
        # 简化：返回所有动作
        return list(range(self.agent.n_actions))


# =====================================================================
#   Phase 3 适配器 - 使用 CurriculumScheduler
# =====================================================================

class Phase3AdaptedTrainer:
    """
    适配课程学习的 RL 训练到三阶段系统
    """

    def __init__(self, env, agent, output_dir: str, config: Optional[Dict] = None):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        default_config = {
            "episodes": 2000,
            "start_epsilon": 0.2,
            "max_steps_per_episode": 50,
            "eval_every": 100,
            "eval_episodes": 20,
            "save_every": 100,
            "use_curriculum": True
        }
        self.cfg = {**default_config, **(config or {})}

        # 课程调度器（如果启用）
        self.curriculum = None
        if self.cfg["use_curriculum"] and hasattr(env, 'request_list'):
            try:
                self.curriculum = CurriculumScheduler(env.request_list)
                logger.info("✓ 启用课程学习")
            except Exception as e:
                logger.warning(f"课程学习初始化失败: {e}")

        # 统计
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "acceptance_rates": [],
            "eval_rewards": [],
            "best_eval_reward": -float('inf')
        }

    def run(self) -> Dict[str, Any]:
        """运行 RL 微调"""
        logger.info("=" * 70)
        logger.info("Phase 3: 强化学习微调 (使用 Curriculum Learning)")
        logger.info("=" * 70)

        # 切换到 RL 模式
        self.agent.switch_to_rl_mode(start_epsilon=self.cfg["start_epsilon"])

        # 重置环境统计
        if hasattr(self.env, 'reset_all'):
            self.env.reset_all()

        for ep in range(1, self.cfg["episodes"] + 1):
            # 获取当前 epsilon
            if self.curriculum:
                epsilon = self.curriculum.get_epsilon()
                expert_ratio = self.curriculum.get_expert_ratio()
            else:
                epsilon = self.agent.get_epsilon()
                expert_ratio = 0.0

            # 运行一个 episode
            ep_reward, ep_length = self._run_episode(epsilon, expert_ratio)

            # 记录统计
            self.stats["episode_rewards"].append(ep_reward)
            self.stats["episode_lengths"].append(ep_length)

            # 计算接受率
            if hasattr(self.env, 'total_requests_seen'):
                acc_rate = (self.env.total_requests_accepted /
                            max(1, self.env.total_requests_seen) * 100)
                self.stats["acceptance_rates"].append(acc_rate)
            else:
                acc_rate = 0.0

            # 课程进度
            if self.curriculum:
                self.curriculum.step()

            # 定期日志
            if ep % 10 == 0:
                recent_rewards = self.stats["episode_rewards"][-10:]
                avg_reward = np.mean(recent_rewards)

                logger.info(
                    f"Episode {ep}/{self.cfg['episodes']} | "
                    f"Reward: {ep_reward:.2f} (avg: {avg_reward:.2f}) | "
                    f"Steps: {ep_length} | Acc: {acc_rate:.2f}% | "
                    f"ε: {epsilon:.3f}"
                )

                if self.curriculum:
                    stage = self.curriculum.get_current_stage()
                    logger.info(f"  课程阶段: {stage.name}")

            # 定期评估
            if ep % self.cfg["eval_every"] == 0:
                eval_reward = self._evaluate()
                self.stats["eval_rewards"].append(eval_reward)

                logger.info(f"[评估] Episode {ep}: {eval_reward:.2f}")

                # 保存最佳模型
                if eval_reward > self.stats["best_eval_reward"]:
                    self.stats["best_eval_reward"] = eval_reward
                    self.agent.save(str(self.output_dir / "rl_best.pth"))
                    logger.info(f"✓ 新最佳模型 (评估奖励: {eval_reward:.2f})")

            # 定期保存
            if ep % self.cfg["save_every"] == 0:
                self.agent.save(str(self.output_dir / f"rl_ep{ep}.pth"))

        # 保存最终模型
        self.agent.save(str(self.output_dir / "rl_final.pth"))

        # 保存统计
        with open(self.output_dir / "phase3_stats.pkl", "wb") as f:
            pickle.dump(self.stats, f)

        # 汇总结果
        result = {
            "avg_reward": np.mean(self.stats["episode_rewards"]),
            "final_acceptance_rate": (self.stats["acceptance_rates"][-1]
                                      if self.stats["acceptance_rates"] else 0),
            "best_eval_reward": self.stats["best_eval_reward"]
        }

        logger.info("=" * 70)
        logger.info("Phase 3 完成")
        logger.info(f"平均奖励: {result['avg_reward']:.2f}")
        logger.info(f"最终接受率: {result['final_acceptance_rate']:.2f}%")
        logger.info(f"最佳评估: {result['best_eval_reward']:.2f}")
        logger.info("=" * 70)

        return result

    def _run_episode(self, epsilon, expert_ratio) -> Tuple[float, int]:
        try:
            req, state = self.env.reset_request()
        except Exception as e:
            logger.error(f"重置失败: {e}")
            return 0.0, 0

        if req is None:
            return 0.0, 0

        done = False
        episode_reward = 0.0
        step = 0
        max_steps = self.cfg["max_steps_per_episode"]

        while not done and step < max_steps:

            if not hasattr(self.env, 'unadded_dest_indices'):
                done = True
                continue

            candidates = list(self.env.unadded_dest_indices)
            if not candidates:
                # ❗关键：不能 break
                done = True
                continue

            goal = candidates[0]

            try:
                valid_actions = self.env.get_valid_low_level_actions()
            except:
                valid_actions = list(range(self.agent.n_actions))

            if not valid_actions:
                valid_actions = [0]

            try:
                action = self.agent.select_action(
                    state, goal, valid_actions, epsilon=epsilon
                )
            except:
                action = valid_actions[0]

            try:
                next_state, reward, sub_done, req_done = self.env.step_low_level(goal, action)
            except Exception as e:
                logger.error(f"环境步进失败: {e}")
                done = True
                continue

            # 🔥 reward 一定在这里被累加
            episode_reward += reward
            step += 1
            state = next_state

            if req_done:
                done = True

        return episode_reward, step

    def _evaluate(self) -> float:
        """评估当前策略"""

        eval_rewards = []

        for _ in range(self.cfg["eval_episodes"]):

            # ================================
            # [MOD-3] reset 失败直接跳过
            # ================================
            try:
                req, state = self.env.reset_request()
            except Exception:
                continue

            if req is None:
                continue

            done = False
            episode_reward = 0.0
            step = 0

            # ================================
            # 主评估循环（与训练一致）
            # ================================
            while not done and step < self.cfg["max_steps_per_episode"]:

                if not hasattr(self.env, "unadded_dest_indices"):
                    break

                candidates = list(self.env.unadded_dest_indices)
                if not candidates:
                    break

                goal = candidates[0]

                try:
                    valid_actions = self.env.get_valid_low_level_actions()
                except Exception:
                    valid_actions = list(range(self.agent.n_actions))

                if not valid_actions:
                    break

                try:
                    action = self.agent.select_action(
                        state, goal, valid_actions, epsilon=0.0
                    )
                except Exception:
                    action = valid_actions[0]

                try:
                    next_state, reward, sub_done, req_done = self.env.step_low_level(
                        goal, action
                    )
                except Exception:
                    break

                state = next_state
                episode_reward += reward
                step += 1

                if req_done:
                    done = True

            eval_rewards.append(episode_reward)

        return float(np.mean(eval_rewards)) if eval_rewards else 0.0


# =====================================================================
#   主训练器 - 整合三个阶段
# =====================================================================

class HIRLSFCThreePhaseTrainer:
    """
    三阶段 HIRL-SFC 训练器
    整合你的所有现有模块
    """

    def __init__(self, env, agent, work_dir: str = None, config: Optional[Dict] = None):
        self.env = env
        self.agent = agent

        if work_dir is None:
            work_dir = str(H.OUTPUT_DIR / "three_phase_training")

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        default_config = {
            "phase1": {
                "episodes": 1000,
                "min_confidence": 0.5,
                "save_every": 100
            },
            "phase2": {
                "epochs": 10,
                "batch_size": 32,
                "save_every_epoch": 2,
                "validation_split": 0.1
            },
            "phase3": {
                "episodes": 2000,
                "start_epsilon": 0.2,
                "max_steps_per_episode": 50,
                "eval_every": 100,
                "eval_episodes": 20,
                "save_every": 100,
                "use_curriculum": True
            }
        }

        if config is None:
            self.cfg = default_config
        else:
            self.cfg = {}
            for phase in ["phase1", "phase2", "phase3"]:
                self.cfg[phase] = {**default_config[phase], **config.get(phase, {})}

    def run_three_phase(self) -> Dict[str, Any]:
        """运行完整的三阶段训练"""
        logger.info("=" * 70)
        logger.info(" " * 20 + "三阶段 HIRL-SFC 训练")
        logger.info("=" * 70)

        results = {}

        try:
            # ==================== Phase 1: 专家轨迹采集 ====================
            logger.info(">>> [Phase 1] Loading Phase 1 Dataset...")

            # 🔥 修复：使用正确的数据路径
            import pickle
            from pathlib import Path

            # 数据目录
            data_dir = Path("E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE\generate_requests_depend_on_poisson\data_output")

            phase1_req_file = data_dir / "phase1_requests.pkl"
            phase1_evt_file = data_dir / "phase1_events.pkl"

            logger.info(f"[Phase 1] 数据目录: {data_dir.absolute()}")
            logger.info(f"[Phase 1] 检查数据文件：")
            logger.info(f"  - requests.pkl: {'存在' if phase1_req_file.exists() else '不存在'}")
            logger.info(f"  - events.pkl: {'存在' if phase1_evt_file.exists() else '不存在'}")

            if phase1_req_file.exists() and phase1_evt_file.exists():
                try:
                    with open(phase1_req_file, 'rb') as f:
                        requests = pickle.load(f)
                    with open(phase1_evt_file, 'rb') as f:
                        events = pickle.load(f)

                    # 设置环境数据
                    self.env.request_list = requests
                    self.env.event_list = events
                    logger.info(f"✓ Loaded Phase 1 data: {len(requests)} requests, {len(events)} events")

                    # 🔥 记录第一个请求信息（用于后续对比）
                    if len(requests) > 0:
                        first_req = requests[0]
                        self._phase1_first_request_id = first_req.get('id', 'N/A')
                        logger.info(f"[Phase 1] 第一个请求 ID: {self._phase1_first_request_id}")
                        logger.info(f"[Phase 1] 第一个请求 Source: {first_req.get('source', 'N/A')}")
                        logger.info(f"[Phase 1] 第一个请求 Dest: {first_req.get('dest', 'N/A')}")

                except Exception as e:
                    logger.error(f"Failed to load Phase 1 data: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                logger.error(f"Phase 1 data files not found in {data_dir.absolute()}")
                raise FileNotFoundError(f"Missing data files in {data_dir}")

            p1_dir = self.work_dir / "phase1"
            phase1 = Phase1AdaptedCollector(self.env, str(p1_dir), self.cfg["phase1"])
            expert_data = phase1.run()

            results["phase1"] = {
                "num_transitions": len(expert_data)
            }

            if not expert_data:
                raise ValueError("Phase 1 failed: No expert data collected!")

            # ==================== Phase 2: 监督学习 ====================
            logger.info(">>> [Phase 2] Starting Supervised Learning...")
            # Phase 2 不需要切换数据集（使用 Phase 1 采集的 expert_data）

            p2_dir = self.work_dir / "phase2"
            phase2 = Phase2AdaptedTrainer(self.agent, expert_data, str(p2_dir), self.cfg["phase2"])
            phase2.run()

            results["phase2"] = {
                "history": phase2.history
            }

            # ==================== Phase 3: RL 微调 ====================
            logger.info("=" * 70)
            logger.info(">>> [Phase 3] Loading Phase 3 Dataset (Harder/Longer)...")
            logger.info("=" * 70)

            # 🔥 诊断：记录加载前的数据状态
            logger.info(f"[Phase 3] 数据加载前状态：")
            if hasattr(self.env, 'request_list'):
                logger.info(f"  - request_list 长度: {len(self.env.request_list)}")
                if len(self.env.request_list) > 0:
                    first_req = self.env.request_list[0]
                    logger.info(f"  - 第一个请求 ID: {first_req.get('id', 'N/A')}")
            else:
                logger.info(f"  - 环境没有 request_list 属性")

            if hasattr(self.env, 'event_list'):
                logger.info(f"  - event_list 长度: {len(self.env.event_list)}")
            else:
                logger.info(f"  - 环境没有 event_list 属性")

            # 🔥 修复：使用正确的数据路径加载 Phase 3 数据
            # 如果有单独的 phase3 数据文件，使用它；否则使用相同的数据
            phase3_req_file = data_dir / "phase3_requests.pkl"
            phase3_evt_file = data_dir / "phase3_events.pkl"

            # 如果没有单独的 phase3 文件，使用默认的 requests.pkl
            if not phase3_req_file.exists():
                logger.info(f"[Phase 3] 未找到 phase3_requests.pkl，使用 requests.pkl")
                phase3_req_file = data_dir / "requests.pkl"
                phase3_evt_file = data_dir / "events.pkl"

            logger.info(f"[Phase 3] 数据目录: {data_dir.absolute()}")
            logger.info(f"[Phase 3] 检查数据文件：")
            logger.info(f"  - {phase3_req_file.name}: {'存在' if phase3_req_file.exists() else '不存在'}")
            logger.info(f"  - {phase3_evt_file.name}: {'存在' if phase3_evt_file.exists() else '不存在'}")

            if phase3_req_file.exists() and phase3_evt_file.exists():
                try:
                    logger.info(f"[Phase 3] 开始加载 Phase 3 数据...")

                    with open(phase3_req_file, 'rb') as f:
                        requests = pickle.load(f)
                    logger.info(f"  ✓ 成功加载 requests: {len(requests)} 个")

                    with open(phase3_evt_file, 'rb') as f:
                        events = pickle.load(f)
                    logger.info(f"  ✓ 成功加载 events: {len(events)} 个")

                    # 显示第一个请求的详细信息
                    if len(requests) > 0:
                        first_req = requests[0]
                        logger.info(f"[Phase 3] Phase 3 数据集第一个请求：")
                        logger.info(f"  - ID: {first_req.get('id', 'N/A')}")
                        logger.info(f"  - Source: {first_req.get('source', 'N/A')}")
                        logger.info(f"  - Dest: {first_req.get('dest', 'N/A')}")
                        logger.info(f"  - VNF 数量: {len(first_req.get('vnf', []))}")

                    # 设置环境数据
                    logger.info(f"[Phase 3] 设置环境数据...")
                    self.env.request_list = requests
                    self.env.event_list = events

                    # 验证设置成功
                    if hasattr(self.env, 'request_list'):
                        logger.info(f"  ✓ 环境 request_list 已更新: {len(self.env.request_list)} 个")

                    logger.info(f"[Phase 3] ✅ Phase 3 数据加载成功！")

                except Exception as e:
                    logger.error(f"[Phase 3] ❌ 加载 Phase 3 数据失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                logger.error(f"[Phase 3] ❌ Phase 3 数据文件不存在于 {data_dir.absolute()}")
                raise FileNotFoundError(f"Missing Phase 3 data files in {data_dir}")

            # 🔥 诊断：记录最终使用的数据状态
            logger.info(f"[Phase 3] 最终使用的数据状态：")
            if hasattr(self.env, 'request_list'):
                logger.info(f"  - request_list 长度: {len(self.env.request_list)}")
                if len(self.env.request_list) > 0:
                    first_req = self.env.request_list[0]
                    logger.info(f"  - 第一个请求 ID: {first_req.get('id', 'N/A')}")
                    logger.info(f"  - 第一个请求 Source: {first_req.get('source', 'N/A')}")
                    logger.info(f"  - 第一个请求 Dest: {first_req.get('dest', 'N/A')}")

            # 🔥 额外检查：验证数据是否真的不同
            if hasattr(self, '_phase1_first_request_id'):
                current_first_id = self.env.request_list[0].get('id', 'N/A') if hasattr(self.env,
                                                                                        'request_list') and len(
                    self.env.request_list) > 0 else 'N/A'
                if current_first_id == self._phase1_first_request_id:
                    logger.warning(f"[Phase 3] ⚠️⚠️⚠️  警告：Phase 3 使用的数据与 Phase 1 相同！")
                    logger.warning(f"[Phase 3] 第一个请求 ID 相同: {current_first_id}")
                else:
                    logger.info(f"[Phase 3] ✅ 确认：Phase 3 使用的是不同的数据集")
                    logger.info(f"[Phase 3] Phase 1 第一个请求 ID: {self._phase1_first_request_id}")
                    logger.info(f"[Phase 3] Phase 3 第一个请求 ID: {current_first_id}")

            logger.info("=" * 70)

            # 🔥 重要：重置环境，确保 Phase 3 能正常运行
            logger.info("[Phase 3] 重置环境...")
            if hasattr(self.env, 'reset_all'):
                self.env.reset_all()
                logger.info("[Phase 3] ✓ 环境已重置（reset_all）")

            # 测试 reset_request 是否正常
            logger.info("[Phase 3] 测试 reset_request...")
            try:
                test_req, test_state = self.env.reset_request()
                if test_req is None:
                    logger.error("[Phase 3] ❌ reset_request 返回 None！")
                    logger.error("[Phase 3] 环境可能没有正确初始化")
                else:
                    logger.info(f"[Phase 3] ✓ reset_request 正常，请求 ID: {test_req.get('id', 'N/A')}")
            except Exception as e:
                logger.error(f"[Phase 3] ❌ reset_request 失败: {e}")
                import traceback
                traceback.print_exc()

            logger.info("=" * 70)

            p3_dir = self.work_dir / "phase3"
            phase3 = Phase3AdaptedTrainer(self.env, self.agent, str(p3_dir), self.cfg["phase3"])
            phase3_stats = phase3.run()

            results["phase3"] = phase3_stats

        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 保存完整结果
        with open(self.work_dir / "training_results.pkl", "wb") as f:
            pickle.dump(results, f)

        logger.info("=" * 70)
        logger.info(" " * 20 + "训练完成！")
        logger.info("=" * 70)
        logger.info(f"Phase 1: {results['phase1']['num_transitions']} transitions")
        logger.info(f"Phase 3: 平均奖励 {results['phase3']['avg_reward']:.2f}")
        logger.info(f"Phase 3: 最佳评估 {results['phase3']['best_eval_reward']:.2f}")
        logger.info(f"输出目录: {self.work_dir}")
        logger.info("=" * 70)

        return results


# =====================================================================
#   主函数
# =====================================================================

def main():
    """主训练函数"""

    print("=" * 70)
    print("三阶段 HIRL-SFC 训练系统")
    print("=" * 70)

    # ========== 1. 创建环境 ==========
    print("\n[1/3] 创建环境...")

    try:
        env = SFC_HIRL_Env_GNN(
            input_dir=H.INPUT_DIR,
            topo=H.TOPOLOGY_MATRIX,
            dc_nodes=H.DC_NODES,
            capacities=H.CAPACITIES,
            use_gnn=True
        )

        print(f"✓ 环境创建成功")
        print(f"  - 节点数: {env.n}")
        print(f"  - 节点特征维度: {env.node_feat_dim}")
        print(f"  - 边特征维度: {env.edge_feat_dim}")
        print(f"  - 请求维度: {env.request_dim}")

    except Exception as e:
        logger.error(f"环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 2. 创建 Agent ==========
    print("\n[2/3] 创建 Agent...")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = MulticastGATWrapperVectorized(
            node_feat_dim=env.node_feat_dim,
            edge_feat_dim=env.edge_feat_dim,
            request_dim=env.request_dim,
            n_actions=env.n,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            tree_pooling='attention'
        ).to(device)

        agent = Agent_SFC_GNN(
            #env=env,  # ✅ 添加这行（第一个参数）
            model=model,
            n_actions=env.n,
            lr=H.LR,
            gamma=H.GAMMA,
            buffer_size=H.EXP_MEMORY,
            batch_size=H.BATCH_SIZE,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"✓ Agent 创建成功")
        print(f"  - 动作空间: {env.n}")
        print(f"  - 设备: {agent.device}")

    except Exception as e:
        logger.error(f"Agent 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 3. 开始训练 ==========
    print("\n[3/3] 开始训练...")

    # 训练配置
    config = {
        "phase1": {
            "episodes": 500,  # 根据需要调整
            "min_confidence": 0.5,
            "save_every": 100
        },
        "phase2": {
            "epochs": 10,
            "batch_size": 32,
        },
        "phase3": {
            "episodes": 1000,
            "start_epsilon": 0.2,
            "use_curriculum": False

        }
    }

    try:
        trainer = HIRLSFCThreePhaseTrainer(
            env=env,
            agent=agent,
            config=config
        )

        results = trainer.run_three_phase()

        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"\n结果:")
        print(f"  Phase 1: {results['phase1']['num_transitions']} 个转换")
        print(f"  Phase 2: 最终损失 {results['phase2']['history']['train_loss'][-1]:.4f}")
        print(f"  Phase 3: 平均奖励 {results['phase3']['avg_reward']:.2f}")
        print(f"  Phase 3: 最佳评估 {results['phase3']['best_eval_reward']:.2f}")
        print(f"  Phase 3: 最终接受率 {results['phase3']['final_acceptance_rate']:.2f}%")
        print("=" * 70)

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()