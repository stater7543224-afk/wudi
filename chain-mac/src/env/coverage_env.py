import numpy as np
from typing import Tuple


class CooperativeCoverageEnv:
    """Multi-agent cooperative coverage environment.

    - 6 agents in 2D continuous space [-1, 1]^2
    - 3 fixed landmarks at positions [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]
    - 3 agent types (fast, balanced, strong), two of each
    - Reward: negative mean distance to nearest uncovered landmark
    - Episode ends when all landmarks covered or max_steps reached
    """

    LANDMARKS = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], dtype=np.float32)
    COVERAGE_RADIUS = 0.3
    AGENT_SPEEDS = np.array([1.5, 1.5, 1.0, 1.0, 0.7, 0.7], dtype=np.float32)
    AGENT_TYPES = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)  # 0=fast, 1=balanced, 2=strong
    ACTION_SPACE = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]], dtype=np.float32)  # N, S, E, W, stay

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.n_agents = len(self.AGENT_SPEEDS)
        self.n_landmarks = len(self.LANDMARKS)
        self.n_agent_types = int(self.AGENT_TYPES.max()) + 1
        self.step_count = 0
        self.agents_pos = None
        self.covered = None

    def reset(self) -> np.ndarray:
        """Reset environment. Returns global state [n_agents, obs_dim]."""
        rng = np.random.default_rng()
        self.agents_pos = rng.uniform(-0.8, 0.8, (self.n_agents, 2)).astype(np.float32)
        self.step_count = 0
        self.covered = np.zeros(self.n_landmarks, dtype=bool)
        return self._get_state()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take a step. actions shape [n_agents], each in {0,1,2,3,4}."""
        self.step_count += 1

        # Move agents
        for i in range(self.n_agents):
            move = self.ACTION_SPACE[actions[i]] * self.AGENT_SPEEDS[i] * 0.1
            self.agents_pos[i] = np.clip(self.agents_pos[i] + move, -1.0, 1.0)

        # Update coverage
        for l_idx in range(self.n_landmarks):
            dists = np.linalg.norm(self.agents_pos - self.LANDMARKS[l_idx], axis=1)
            self.covered[l_idx] = np.any(dists < self.COVERAGE_RADIUS)

        # Reward
        all_covered = self.covered.all()
        if all_covered:
            reward = 10.0
        else:
            uncovered_dists = []
            for l_idx in range(self.n_landmarks):
                if not self.covered[l_idx]:
                    dists = np.linalg.norm(self.agents_pos - self.LANDMARKS[l_idx], axis=1)
                    uncovered_dists.append(dists.min())
            reward = float(-np.mean(uncovered_dists)) if uncovered_dists else 0.0

        done = bool(all_covered or (self.step_count >= self.max_steps))
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        """Return per-agent observations [n_agents, obs_dim].

        obs = [x, y, type_onehot(3), dist_to_landmarks(3)]
        """
        batch = []
        for i in range(self.n_agents):
            pos = self.agents_pos[i]
            type_oh = np.zeros(self.n_agent_types, dtype=np.float32)
            type_oh[self.AGENT_TYPES[i]] = 1.0
            dists = np.linalg.norm(self.agents_pos[i] - self.LANDMARKS, axis=1).astype(np.float32)
            obs = np.concatenate([pos, type_oh, dists])
            batch.append(obs)
        return np.stack(batch, axis=0)

    def render(self):
        """Simple text render."""
        print(f"Step {self.step_count}, Agents: {self.agents_pos.round(2)}")
        print(f"Covered: {self.covered}, Reward: {self._get_reward():.3f}")

    def _get_reward(self) -> float:
        if self.covered.all():
            return 10.0
        uncovered_dists = []
        for l_idx in range(self.n_landmarks):
            if not self.covered[l_idx]:
                dists = np.linalg.norm(self.agents_pos - self.LANDMARKS[l_idx], axis=1)
                uncovered_dists.append(dists.min())
        return float(-np.mean(uncovered_dists)) if uncovered_dists else 0.0
