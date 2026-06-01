import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixNet(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim, state_dim,
                 rnn_hidden=64, mixing_hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

        # agent RNN
        self.agent_rnn = nn.GRUCell(obs_dim + action_dim, rnn_hidden)
        self.agent_fc = nn.Linear(rnn_hidden, action_dim)

        # hypernets for mixing
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden * n_agents),
            nn.ReLU(),
            nn.Linear(mixing_hidden * n_agents, mixing_hidden * n_agents),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1),
        )

    def get_individual_qs(self, obs, prev_actions, hidden_states):
        """Returns q_values [B, n_agents, n_actions], new_hidden [B, n_agents, rnn_hidden]."""
        batch, n_agents, _ = obs.shape
        rnn_in = torch.cat([obs, prev_actions], dim=-1).view(batch * n_agents, -1)
        h = hidden_states.view(batch * n_agents, -1)
        new_h = self.agent_rnn(rnn_in, h)
        q = self.agent_fc(new_h)
        return q.view(batch, n_agents, self.action_dim), new_h.view(batch, n_agents, -1)

    def mix(self, q_values, states):
        """Mix [B, n_agents, 1] Qs into [B, 1] total Q using hypernets."""
        batch = q_values.shape[0]
        w1 = self.hyper_w1(states).view(batch, self.n_agents, -1)
        b1 = self.hyper_b1(states).view(batch, 1, -1)
        hidden = F.elu(torch.bmm(q_values.transpose(1, 2), w1) + b1)

        w2 = self.hyper_w2(states).view(batch, -1, 1)
        b2 = self.hyper_b2(states).view(batch, 1, 1)
        return torch.bmm(hidden, w2) + b2


class QMIX(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim, state_dim,
                 rnn_hidden=64, mixing_hidden=64,
                 lr=3e-4, gamma=0.99, device="cpu"):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        self.q_net = QMixNet(n_agents, obs_dim, action_dim, state_dim,
                             rnn_hidden, mixing_hidden)
        self.target_net = QMixNet(n_agents, obs_dim, action_dim, state_dim,
                                  rnn_hidden, mixing_hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def _to_onehot(self, actions):
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    def update(self, states, actions, rewards, next_states, dones,
               hidden_states=None, global_states=None, next_global_states=None):
        batch = states.shape[0]

        if hidden_states is None:
            hidden_states = torch.zeros(batch, self.n_agents, 64, device=self.device)

        prev_actions = self._to_onehot(actions)

        # current Q
        q_vals, _ = self.q_net.get_individual_qs(states, prev_actions, hidden_states)
        chosen_q = q_vals.gather(2, actions.unsqueeze(-1).long()).squeeze(-1)

        # target Q (max over actions)
        with torch.no_grad():
            target_q_vals, _ = self.target_net.get_individual_qs(
                next_states, prev_actions, hidden_states)
            target_max_q = target_q_vals.max(dim=-1, keepdim=True)[0]

        # mixing
        if global_states is not None:
            s_mix = global_states
            ns_mix = next_global_states if next_global_states is not None else global_states
        else:
            s_mix = states.reshape(batch, -1)
            ns_mix = next_states.reshape(batch, -1)

        total_q = self.q_net.mix(chosen_q.unsqueeze(-1), s_mix)
        with torch.no_grad():
            total_target = self.target_net.mix(target_max_q, ns_mix)

        # TD loss
        targets = rewards + self.gamma * (1 - dones.float()) * total_target
        loss = F.mse_loss(total_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
