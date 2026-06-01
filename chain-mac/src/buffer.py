import numpy as np
from typing import Optional


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int,
                 state_dim: Optional[int] = None):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.has_global = state_dim is not None and state_dim > 0

        self.states = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=bool)

        if self.has_global:
            self.global_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_global_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state, done,
             global_state=None, next_global_state=None):
        idx = self.pos % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        if self.has_global:
            self.global_states[idx] = global_state
            self.next_global_states[idx] = next_global_state
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        batch = (self.states[idx], self.actions[idx], self.rewards[idx],
                 self.next_states[idx], self.dones[idx])
        if self.has_global:
            batch += (self.global_states[idx], self.next_global_states[idx])
        return batch

    def __len__(self):
        return self.size
