import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiffusionPolicy(nn.Module):
    """Continuous denoising for discrete action generation.

    Forward: add noise to one-hot actions.
    Reverse: denoise from Gaussian to action logits.
    Has optional global context for cross-group coordination.
    """

    def __init__(self, obs_dim: int, action_dim: int, n_steps: int = 10,
                 hidden_dim: int = 256, device: str = "cpu",
                 global_dim: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.device = device
        self.global_dim = global_dim

        denoise_in = action_dim + obs_dim + global_dim + 1
        self.denoise_net = nn.Sequential(
            nn.Linear(denoise_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        self.to(device)

    def forward(self, obs: torch.Tensor,
                global_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample actions: reverse diffusion from N(0,I)."""
        B = obs.shape[0]
        x = torch.randn(B, self.action_dim, device=self.device)

        for t in reversed(range(self.n_steps)):
            t_t = torch.full((B, 1), t / self.n_steps, device=self.device)
            inp = [x, obs, t_t]
            if self.global_dim > 0:
                inp.append(global_ctx if global_ctx is not None
                          else torch.zeros(B, self.global_dim, device=self.device))
            pred = self.denoise_net(torch.cat(inp, dim=-1))

            if t > 0:
                a = self.alphas[t]
                ab = self.alpha_bars[t]
                b = self.betas[t]
                noise = torch.randn_like(x)
                x = (x - (1 - a) / torch.sqrt(1 - ab) * pred) / torch.sqrt(a)
                x = x + torch.sqrt(b) * noise
            else:
                x = x - pred

        return x  # action logits

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                     global_ctx: Optional[torch.Tensor] = None,
                     return_logits: bool = False):
        """Diffusion MSE loss, optionally with one-step denoised logits for Q-guidance."""
        B = obs.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device)

        noise = torch.randn_like(actions)
        ab = self.alpha_bars[t].view(-1, 1)
        noisy = torch.sqrt(ab) * actions + torch.sqrt(1 - ab) * noise

        t_t = t.float().view(-1, 1) / self.n_steps
        inp = [noisy, obs, t_t]
        if self.global_dim > 0:
            inp.append(global_ctx if global_ctx is not None
                      else torch.zeros(B, self.global_dim, device=self.device))
        noise_pred = self.denoise_net(torch.cat(inp, dim=-1))

        loss = F.mse_loss(noise_pred, noise)

        if return_logits:
            denoised = (noisy - torch.sqrt(1 - ab) * noise_pred) / torch.sqrt(ab)
            return loss, denoised

        return loss

    def to(self, device):
        super().to(device)
        self.device = device
        return self
