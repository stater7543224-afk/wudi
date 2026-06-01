import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.cluster import KMeans


class RewardModel(nn.Module):
    """Reward predictor r^(a, s) for counterfactual estimation."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action_onehot], dim=-1))


class CausalPerception:
    """ACE-based causal perception.

    ACE(S^i) = sum_{s in vals(S^i)} KL(P(r) || P(r | do(S^i = s)))
    w_i = softmax(ACE)_i

    Maintains FIFO (s,a,r) buffer, trains a reward model,
    enumerates subspace values via KMeans, measures KL divergence,
    returns softmax-normalised subspace importance weights.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 subspace_indices: list, device: str = "cpu"):
        self.subspaces = subspace_indices
        self.n_subspaces = len(subspace_indices)
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardModel(state_dim, action_dim).to(device)
        self.reward_optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=1e-3)

        self.prior_action_dist = np.ones(action_dim, dtype=np.float32) / action_dim
        self.weights = np.ones(self.n_subspaces, dtype=np.float32) / self.n_subspaces

        self._buf_s = deque(maxlen=3000)
        self._buf_a = deque(maxlen=3000)
        self._buf_r = deque(maxlen=3000)

    # --- internal helpers -------------------------------------------------

    def _push_experiences(self, states, actions, rewards):
        if states.ndim == 3:
            B, N, D = states.shape
            s_flat = states.reshape(B * N, D)
            a_flat = actions.reshape(B * N)
            r_flat = np.repeat(rewards, N) if rewards.ndim > 1 else np.repeat(rewards, N)
        else:
            s_flat = states
            a_flat = actions.flatten() if actions.ndim > 1 else actions
            r_flat = rewards.flatten() if rewards.ndim > 1 else rewards

        for i in range(len(s_flat)):
            self._buf_s.append(s_flat[i])
            self._buf_a.append(a_flat[i])
            self._buf_r.append(r_flat[i])

    def _train_reward_model(self, batch_size=64, epochs=30):
        if len(self._buf_r) < batch_size * 2:
            return

        states = np.array(self._buf_s, dtype=np.float32)
        actions = np.array(self._buf_a, dtype=np.int64)
        rewards = np.array(self._buf_r, dtype=np.float32)

        N = len(states)
        act_onehot = np.zeros((N, self.action_dim), dtype=np.float32)
        act_onehot[np.arange(N), actions] = 1.0

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(states), torch.from_numpy(act_onehot),
            torch.from_numpy(rewards).view(-1, 1))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                              shuffle=True, drop_last=True)

        self.reward_model.train()
        for _ in range(epochs):
            for s, a, r in loader:
                s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
                loss = nn.MSELoss()(self.reward_model(s, a), r)
                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()

    def _get_possible_values(self, states, subspace_idx, max_vals=10):
        sub_data = states[:, self.subspaces[subspace_idx]]
        k = min(max_vals, max(2, len(sub_data) // 10))
        km = KMeans(n_clusters=k, n_init=3, random_state=0)
        km.fit(sub_data)
        return km.cluster_centers_

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
        return (np.log(sigma_p / sigma_q)
                + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5)

    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float64)
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)

    # --- public API -------------------------------------------------------

    def compute_importance(self, states, actions, rewards):
        """Compute causal importance weights (ACE-based).

        Returns softmax-normalised per-subspace importance.
        """
        self._push_experiences(states, actions, rewards)
        if len(self._buf_r) < 32:
            self.weights = np.ones(self.n_subspaces, dtype=np.float32) / self.n_subspaces
            return self.weights
        self._train_reward_model()

        B, N, D = states.shape
        flat_s = states.reshape(B * N, D)
        flat_r = np.repeat(rewards, N) if rewards.ndim > 1 else np.repeat(rewards, N)

        mu_orig = float(np.mean(flat_r))
        sigma_orig = float(np.std(flat_r)) + 1e-8

        ace = np.zeros(self.n_subspaces, dtype=np.float64)

        for idx, subspace in enumerate(self.subspaces):
            s_vals = self._get_possible_values(flat_s, idx)
            total_kl = 0.0

            for s_val in s_vals:
                s_star = flat_s.copy()
                s_star[:, subspace] = s_val

                n_flat = len(flat_s)
                s_batch = np.repeat(s_star, self.action_dim, axis=0)
                a_batch = np.zeros((n_flat * self.action_dim, self.action_dim),
                                   dtype=np.float32)
                for a in range(self.action_dim):
                    a_batch[a::self.action_dim, a] = 1.0

                self.reward_model.eval()
                with torch.no_grad():
                    s_t = torch.from_numpy(s_batch).to(self.device)
                    a_t = torch.from_numpy(a_batch).to(self.device)
                    preds = self.reward_model(s_t, a_t).cpu().numpy().flatten()

                mu_intervened = float(np.mean(preds.reshape(n_flat, self.action_dim)
                                              @ self.prior_action_dist))
                kl = self._kl_gaussian(mu_intervened, sigma_orig, mu_orig, sigma_orig)
                total_kl += max(0.0, kl)

            ace[idx] = total_kl / max(1, len(s_vals))

        self.weights = self._softmax(ace)
        return self.weights

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Weight each subspace dim by its importance."""
        w = torch.as_tensor(self.weights, dtype=torch.float32, device=state.device)
        result = state.clone()
        for i, subspace in enumerate(self.subspaces):
            result[..., subspace] *= w[i]
        return result
