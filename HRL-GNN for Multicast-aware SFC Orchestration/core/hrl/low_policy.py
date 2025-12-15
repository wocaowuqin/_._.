#完成部署流程
import torch
import torch.nn as nn

class LowLevelPolicy(nn.Module):
    """
    低层策略网络 (Action Selector)
    对应 Env 中的 Low-Level Path Selection (step_low_level)
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(LowLevelPolicy, self).__init__()

        # 1. Actor: 输出低层动作的概率 (Logits)
        # 这里的动作通常是: Path_Index * K_Path + K_Index
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Output: [B, NB_LOW_LEVEL_ACTIONS]
        )

        # 2. Critic: 评估当前状态在低层的价值
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