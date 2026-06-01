import numpy as np
from src.config import Config


def make_smac_config(map_name="3m", n_groups=3,
                     total_train_steps=2_000_000, **overrides):
    """Auto-detect dims from SMAC and build Config."""
    from smac.env import StarCraft2Env

    boot = StarCraft2Env(map_name=map_name)
    boot.reset()
    n_agents = boot.n_agents
    n_actions = boot.n_actions
    obs_dim = boot.get_obs()[0].shape[0]
    state_dim = boot.get_state().shape[0]
    episode_limit = boot.episode_limit
    boot.close()

    subspaces = _auto_subspaces(obs_dim, n_groups)

    cfg = Config(
        n_agents=n_agents, n_landmarks=0, n_agent_types=0,
        obs_dim=obs_dim, action_dim=n_actions, state_dim=state_dim,
        episode_len=episode_limit,
        subspaces=subspaces,
        n_groups=n_groups,
        hy_gma_hidden=min(64, obs_dim * 2),
        group_momentum=0.99, group_smooth_alpha=0.6,
        q_boost_lambda=0.01,
        global_ctx_dim=obs_dim if n_agents > 3 else 0,
        total_train_steps=total_train_steps,
        learn_freq=min(100, max(50, episode_limit // 2)),
        eval_freq=min(10000, total_train_steps // 20),
        log_freq=100,
        device="cuda",
    )

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def _auto_subspaces(obs_dim, n_groups):
    """Split obs dims into n_groups roughly equal contiguous chunks."""
    if n_groups < 1:
        n_groups = 1
    chunk = max(1, obs_dim // n_groups)
    subspaces = []
    start = 0
    for i in range(n_groups):
        end = obs_dim if i == n_groups - 1 else start + chunk
        if end > start:
            subspaces.append(list(range(start, end)))
            start = end
    # merge last tiny chunk
    if len(subspaces) > 1 and len(subspaces[-1]) < chunk // 2:
        subspaces[-2].extend(subspaces.pop())
    return subspaces
