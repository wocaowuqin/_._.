#中层选择路径import torch
import torch.nn as nn

class MidLevelPolicy(nn.Module):
    """
    中层策略网络 (Goal Selector)
    对应 Env 中的 High-Level Goal Selection (step_high_level)
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(MidLevelPolicy, self).__init__()

        # 1. Actor: 输出选择每个目标的概率 (Logits)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Output: [B, NB_HIGH_LEVEL_GOALS]
        )

        # 2. Critic: 评估当前状态对完成目标的价值
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)           # Output: [B, 1]
        )

    def forward(self, state_emb):
        """
        :param state_emb: 融合后的全局状态向量 [B, Hidden]
        :return: logits, value
        """
        logits = self.actor(state_emb)
        value = self.critic(state_emb)
        return logits, value