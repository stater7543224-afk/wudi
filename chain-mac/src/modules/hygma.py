import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicHypergraphGrouping(nn.Module):
    """Prototype-based agent grouping with temporal smoothing.

    Encodes obs → embeddings, assigns agents to nearest prototype
    (cosine similarity), then enhances Q-values via group-wise convolution.
    Prototypes EMA-updated; temporal smoothing prevents group-flip per step.
    """

    def __init__(self, obs_dim: int, n_groups: int, hidden_dim: int = 64,
                 momentum: float = 0.99, temp: float = 1.0,
                 smooth_alpha: float = 0.6):
        super().__init__()
        self.n_groups = n_groups
        self.momentum = momentum
        self.temp = temp
        self.smooth_alpha = smooth_alpha

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.prototypes = nn.Parameter(torch.randn(n_groups, hidden_dim) * 0.1)
        self._prev_scores = None

    def compute_groups(self, states: torch.Tensor,
                       inference: bool = True) -> np.ndarray:
        """Nearest-prototype assignment, optionally temporally smoothed."""
        n_agents = states.shape[0]
        if n_agents <= self.n_groups:
            self._prev_scores = None
            return np.arange(n_agents) % self.n_groups

        features = F.normalize(self.encoder(states), dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        scores = features @ protos.T

        if inference and self._prev_scores is not None:
            prev = self._prev_scores.to(scores.device)
            scores = self.smooth_alpha * scores + (1 - self.smooth_alpha) * prev

        if inference:
            self._prev_scores = scores.detach().cpu()

        return scores.argmax(dim=-1).cpu().numpy()

    def update_prototypes(self, states: torch.Tensor):
        """EMA prototypes toward centroid of assigned agents."""
        flat = states.reshape(-1, states.shape[-1])
        if len(flat) < 2:
            return

        with torch.no_grad():
            features = F.normalize(self.encoder(flat), dim=-1)
            protos = F.normalize(self.prototypes, dim=-1)
            scores = features @ protos.T
            soft = F.softmax(scores / self.temp, dim=-1)

            for g in range(self.n_groups):
                w = soft[:, g:g+1]
                centroid = (w * features).sum(dim=0)
                norm = w.sum() + 1e-8
                new_proto = F.normalize(centroid / norm, dim=0)
                self.prototypes[g] = F.normalize(
                    self.momentum * self.prototypes[g] + (1 - self.momentum) * new_proto, dim=0)

    def reset(self):
        self._prev_scores = None

    def hypergraph_convolve(self, q_values: torch.Tensor,
                            groups: np.ndarray) -> torch.Tensor:
        """Group-wise Q enhancement via within-group mean residual."""
        out = q_values.clone()
        for g in range(self.n_groups):
            mask = (groups == g)
            if mask.sum() > 1:
                group_mean = q_values[mask].mean(dim=0, keepdim=True)
                out[mask] = 0.7 * q_values[mask] + 0.3 * group_mean
        return out
