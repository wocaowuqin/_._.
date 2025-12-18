"""
Agent 初始化包装器
解决 agent_gnn.py 需要 model 和 n_actions 参数的问题
"""

from core.hrl.agent.agent_gnn import Agent_SFC_GNN
from core.hrl.high_policy import HighLevelPolicy


def create_agent_from_config(config):
    """
    从配置创建 Agent，自动处理 model 和 n_actions 参数

    Args:
        config: 配置字典，应包含:
            - gnn: GNN配置 (node_feat_dim, edge_feat_dim, request_feat_dim, hidden_dim)
            - env: 环境配置 (nb_high_level_goals, nb_low_level_actions)
            - training: 训练配置 (learning_rate, gamma, batch_size, buffer_size, etc.)

    Returns:
        Agent_SFC_GNN 实例
    """

    # 1. 提取配置
    gnn_config = config.get('gnn', {})
    env_config = config.get('env', {})
    training_config = config.get('training', {})
    eval_config = config.get('eval', {})
    epsilon_config = config.get('epsilon', {})

    # 2. GNN 模型参数
    node_feat_dim = gnn_config.get('node_feat_dim', 10)
    edge_feat_dim = gnn_config.get('edge_feat_dim', 3)
    request_feat_dim = gnn_config.get('request_feat_dim', 6)
    hidden_dim = gnn_config.get('hidden_dim', 128)
    num_gat_layers = gnn_config.get('num_gat_layers', 3)
    num_heads = gnn_config.get('num_heads', 4)
    use_vectorized = gnn_config.get('use_vectorized', False)

    # 3. 动作空间大小
    nb_high_level_goals = env_config.get('nb_high_level_goals', 10)
    nb_low_level_actions = env_config.get('nb_low_level_actions', 50)
    n_actions = nb_low_level_actions  # Low-level policy 的动作数

    # 4. 训练参数
    learning_rate = training_config.get('learning_rate', 1e-4)
    gamma = training_config.get('gamma', 0.99)
    buffer_size = training_config.get('buffer_size', 10000)
    batch_size = training_config.get('batch_size', 32)
    device = eval_config.get('device', 'cuda')

    # 5. Epsilon 参数
    epsilon_start = epsilon_config.get('initial', 1.0)
    epsilon_end = epsilon_config.get('final', 0.01)
    epsilon_decay = epsilon_config.get('decay_steps', 10000)

    # 6. 创建 GNN 模型
    model = HighLevelPolicy(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        request_feat_dim=request_feat_dim,
        hidden_dim=hidden_dim,
        nb_high_level_goals=nb_high_level_goals,
        nb_low_level_actions=nb_low_level_actions,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        use_vectorized=use_vectorized
    )

    # 7. 创建 Agent
    agent = Agent_SFC_GNN(
        model=model,
        n_actions=n_actions,
        lr=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        prioritized_alpha=training_config.get('prioritized_alpha', 0.6),
        prioritized_beta0=training_config.get('prioritized_beta0', 0.4)
    )

    return agent


# 为了向后兼容，创建一个别名
HRL_DQN_Agent = create_agent_from_config

if __name__ == "__main__":
    # 测试用例
    test_config = {
        'gnn': {
            'node_feat_dim': 10,
            'edge_feat_dim': 3,
            'request_feat_dim': 6,
            'hidden_dim': 128,
            'num_gat_layers': 3,
            'num_heads': 4,
            'use_vectorized': False
        },
        'env': {
            'nb_high_level_goals': 10,
            'nb_low_level_actions': 50
        },
        'training': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 32
        },
        'eval': {
            'device': 'cpu'
        },
        'epsilon': {
            'initial': 1.0,
            'final': 0.01,
            'decay_steps': 10000
        }
    }

    agent = create_agent_from_config(test_config)
    print(f"✅ Agent 创建成功")
    print(f"   动作数: {agent.n_actions}")
    print(f"   设备: {agent.device}")
    print(f"   学习率: {agent.lr}")