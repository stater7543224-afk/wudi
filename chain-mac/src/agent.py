import torch
import torch.nn.functional as F
import numpy as np
from .modules.tmae import CausalPerception
from .modules.hygma import DynamicHypergraphGrouping
from .modules.diffusion_policy import DiffusionPolicy
from .modules.qmix import QMIX


class CHAIN_MAC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tmae = CausalPerception(cfg.obs_dim, cfg.action_dim,
                                     cfg.subspaces, device=str(self.device))

        self.hygma = DynamicHypergraphGrouping(
            cfg.obs_dim, cfg.n_groups, cfg.hy_gma_hidden,
            momentum=cfg.group_momentum, smooth_alpha=cfg.group_smooth_alpha)
        self.hygma.to(self.device)

        self.diffusion_policies = [
            DiffusionPolicy(cfg.obs_dim, cfg.action_dim, cfg.diffusion_steps,
                            cfg.diff_hidden, device=str(self.device),
                            global_dim=cfg.global_ctx_dim)
            for _ in range(cfg.n_groups)
        ]
        self.diff_optimizers = [
            torch.optim.Adam(dp.parameters(), lr=cfg.lr)
            for dp in self.diffusion_policies
        ]

        self.qmix = QMIX(cfg.n_agents, cfg.obs_dim, cfg.action_dim, cfg.state_dim,
                         cfg.qmix_rnn_hidden, cfg.qmix_mixing_hidden,
                         cfg.lr, cfg.gamma, device=str(self.device))
        self.qmix.to(self.device)

        self.step_count = 0
        self.groups = None

    def reset(self):
        self.hygma.reset()

    def select_actions(self, obs: np.ndarray, eps: float = 0.0,
                       avail_mask: np.ndarray = None) -> np.ndarray:
        if np.random.random() < eps:
            if avail_mask is not None:
                acts = np.zeros(self.cfg.n_agents, dtype=np.int64)
                for i in range(self.cfg.n_agents):
                    valid = np.where(avail_mask[i])[0]
                    acts[i] = np.random.choice(valid) if len(valid) > 0 else 0
                return acts
            return np.random.randint(0, self.cfg.action_dim, size=self.cfg.n_agents)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        weighted_obs = self.tmae.apply(obs_t)
        self.groups = self.hygma.compute_groups(weighted_obs, inference=True)

        if self.cfg.global_ctx_dim > 0:
            global_ctx = weighted_obs.mean(dim=0, keepdim=True).expand(self.cfg.n_agents, -1)
        else:
            global_ctx = None

        mask_t = torch.as_tensor(avail_mask, dtype=torch.bool, device=self.device) \
            if avail_mask is not None else None

        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for g in range(self.cfg.n_groups):
            gmask = (self.groups == g)
            if gmask.sum() == 0:
                continue
            group_obs = weighted_obs[gmask]
            with torch.no_grad():
                ctx = global_ctx[gmask] if global_ctx is not None else None
                logits = self.diffusion_policies[g](group_obs, global_ctx=ctx)
                if mask_t is not None:
                    logits[~mask_t[gmask]] = -1e9
                probs = F.softmax(logits, dim=-1)
                gacts = torch.multinomial(probs, num_samples=1).squeeze(-1)
                actions[gmask] = gacts.cpu().numpy()
        return actions

    def update(self, states, actions, rewards, next_states, dones,
               global_states=None, next_global_states=None):
        self.step_count += 1
        batch = states.shape[0]
        device = self.device

        s = torch.as_tensor(states, dtype=torch.float32, device=device)
        a = torch.as_tensor(actions, dtype=torch.long, device=device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=device)
        gs = torch.as_tensor(global_states, dtype=torch.float32, device=device) \
            if global_states is not None else None
        ngs = torch.as_tensor(next_global_states, dtype=torch.float32, device=device) \
            if next_global_states is not None else None

        # QMIX
        qmix_loss = self.qmix.update(s, a, r, ns, d,
                                     global_states=gs, next_global_states=ngs)
        if self.step_count % self.cfg.target_update_freq == 0:
            self.qmix.update_target()

        if self.step_count % self.cfg.causal_update_freq == 0:
            self.tmae.compute_importance(states, actions, rewards)

        # Q-guidance prep
        if self.cfg.q_boost_lambda > 0:
            h_zero = torch.zeros(batch, self.cfg.n_agents, self.cfg.qmix_rnn_hidden, device=device)
            prev_a = F.one_hot(a, num_classes=self.cfg.action_dim).float()
            with torch.no_grad():
                q_all, _ = self.qmix.q_net.get_individual_qs(s, prev_a, h_zero)
                q_all = q_all.detach()

        # Update diffusion policies (per-group)
        total_diff_loss = 0.0
        total_q_guide = 0.0
        with torch.no_grad():
            weighted_s = self.tmae.apply(s)

        global_ctx = weighted_s.mean(dim=1) if self.cfg.global_ctx_dim > 0 else None

        for g in range(self.cfg.n_groups):
            dp = self.diffusion_policies[g]
            opt = self.diff_optimizers[g]

            group_mask = np.zeros((batch, self.cfg.n_agents), dtype=bool)
            for b in range(batch):
                groups_b = self.hygma.compute_groups(weighted_s[b], inference=False)
                group_mask[b] = (groups_b == g)

            if not group_mask.any():
                continue

            flat_mask = group_mask.flatten()
            gobs = weighted_s.reshape(batch * self.cfg.n_agents, -1)[flat_mask]
            gact = a.flatten()[flat_mask]
            if len(gobs) < 2:
                continue

            gact_onehot = F.one_hot(gact, num_classes=self.cfg.action_dim).float()
            ctx = None
            if global_ctx is not None:
                ctx = global_ctx.unsqueeze(1).expand(-1, self.cfg.n_agents, -1).flatten(0, 1)[flat_mask]

            if self.cfg.q_boost_lambda > 0:
                diff_loss, denoised = dp.compute_loss(gobs, gact_onehot, global_ctx=ctx, return_logits=True)
                q_group = q_all.flatten(0, 1)[flat_mask]
                soft = F.softmax(denoised, dim=-1)
                q_guide = (soft * q_group).sum(dim=-1).mean()
                diff_loss -= self.cfg.q_boost_lambda * q_guide
                total_q_guide += q_guide.item()
            else:
                diff_loss = dp.compute_loss(gobs, gact_onehot, global_ctx=ctx)

            total_diff_loss += diff_loss.item()
            opt.zero_grad()
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(dp.parameters(), 5.0)
            opt.step()

        total_diff_loss = total_diff_loss / max(1, self.cfg.n_groups)
        total_q_guide = total_q_guide / max(1, self.cfg.n_groups)
        self.hygma.update_prototypes(weighted_s.reshape(-1, self.cfg.obs_dim))

        return {"qmix_loss": qmix_loss, "diff_loss": total_diff_loss,
                "q_guide": total_q_guide}
