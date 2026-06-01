from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # env
    n_agents: int = 6
    n_landmarks: int = 3
    n_agent_types: int = 3
    obs_dim: int = 8
    action_dim: int = 5
    state_dim: int = 48
    episode_len: int = 50

    # TMAE — per-agent subspace indices
    subspaces: List[List[int]] = field(default_factory=lambda: [[0, 1], [2, 3, 4], [5, 6, 7]])

    # HYGMA
    n_groups: int = 3
    hy_gma_hidden: int = 64
    group_momentum: float = 0.99
    group_smooth_alpha: float = 0.6

    # diffusion
    diffusion_steps: int = 10
    diff_hidden: int = 256

    # QMIX
    qmix_rnn_hidden: int = 64
    qmix_mixing_hidden: int = 64

    # training
    buffer_capacity: int = 50000
    batch_size: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    total_train_steps: int = 200_000
    learn_freq: int = 100
    target_update_freq: int = 200
    causal_update_freq: int = 10_000
    group_update_freq: int = 5000
    eval_freq: int = 5000
    log_freq: int = 100

    # coordination
    q_boost_lambda: float = 0.01   # higher = stronger Q-guidance on diffusion
    global_ctx_dim: int = 0

    device: str = "cuda"
