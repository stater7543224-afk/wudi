import numpy as np
from typing import Tuple, Optional


class SMACEnv:
    """Adapter: SMAC -> CHAIN-MAC."""

    def __init__(self, map_name: str = "3m", seed: Optional[int] = None,
                 difficulty: str = "7", **kwargs):
        self.map_name = map_name
        from smac.env import StarCraft2Env
        self._env = StarCraft2Env(
            map_name=map_name,
            difficulty=difficulty,
            seed=seed,
            reward_scale=False,
            reward_only_positive=False,
            **kwargs,
        )
        self._obs_dim: Optional[int] = None
        self._state_dim: Optional[int] = None
        self.n_agents = self._env.n_agents
        self.n_actions = self._env.n_actions
        self.step_count = 0
        self.episode_limit = self._env.episode_limit

    def reset(self) -> np.ndarray:
        self.step_count = 0
        obs_list, global_state = self._env.reset()
        state = np.asarray(global_state, dtype=np.float32)
        if self._state_dim is None:
            self._state_dim = state.shape[-1]
        obs = np.stack([np.asarray(o, dtype=np.float32) for o in obs_list], axis=0)
        if self._obs_dim is None:
            self._obs_dim = obs.shape[-1]
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        reward, terminated, info = self._env.step(actions.tolist())
        next_obs_list = self._env.get_obs()
        next_obs = np.stack([np.asarray(o, dtype=np.float32) for o in next_obs_list], axis=0)
        done = terminated or self.step_count >= self.episode_limit
        return next_obs, float(reward), done, info

    def get_global_state(self) -> np.ndarray:
        return self._env.get_state().astype(np.float32)

    def get_avail_actions(self) -> np.ndarray:
        avail = self._env.get_avail_actions()
        return np.stack(avail, axis=0)

    @property
    def obs_dim(self) -> int:
        assert self._obs_dim is not None
        return self._obs_dim

    @property
    def state_dim(self) -> int:
        assert self._state_dim is not None
        return self._state_dim

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        self._env.render(mode)
