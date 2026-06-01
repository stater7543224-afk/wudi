import numpy as np
import torch
from collections import deque
from .env.coverage_env import CooperativeCoverageEnv
from .buffer import ReplayBuffer
from .agent import CHAIN_MAC


class Trainer:
    """Training loop for CHAIN-MAC."""

    def __init__(self, config):
        self.cfg = config
        self.env = CooperativeCoverageEnv(max_steps=config.episode_len)
        self.agent = CHAIN_MAC(config)
        self.buffer = ReplayBuffer(config.buffer_capacity, config.n_agents, config.obs_dim)
        self.episode_rewards = deque(maxlen=100)

    def train(self):
        cfg = self.cfg
        env = self.env
        agent = self.agent
        buffer = self.buffer

        state = env.reset()
        agent.reset()
        episode_reward = 0.0
        episode_steps = 0
        best_mean_reward = -float("inf")

        for step in range(1, cfg.total_train_steps + 1):
            # Epsilon-greedy annealing
            eps = max(0.05, 1.0 - step / (cfg.total_train_steps * 0.5))

            # Select actions
            actions = agent.select_actions(state, eps=eps)

            # Environment step
            next_state, reward, done = env.step(actions)

            # Store in buffer
            buffer.push(state, actions, reward, next_state, done)
            episode_reward += reward
            episode_steps += 1

            state = next_state

            # Episode done
            if done:
                self.episode_rewards.append(episode_reward)
                state = env.reset()
                agent.reset()
                episode_reward = 0.0
                episode_steps = 0

            # Training step
            if step % cfg.learn_freq == 0 and len(buffer) >= cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                losses = agent.update(*batch)

                if step % cfg.log_freq == 0:
                    mean_r = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
                    q_boost_str = f" | Q-guide {losses.get('q_guide', 0):.4f}"
                    print(f"Step {step:6d} | Eps {eps:.3f} | "
                          f"Mean R {mean_r:6.2f} | "
                          f"QMIX loss {losses['qmix_loss']:.4f} | "
                          f"Diff loss {losses['diff_loss']:.4f}"
                          f"{q_boost_str if losses.get('q_guide', 0) != 0 else ''}")

            # Evaluation
            if step % cfg.eval_freq == 0:
                eval_reward = self.evaluate(n_episodes=5)
                print(f"\n{'='*40}\nEvaluation at step {step}: mean reward = {eval_reward:.2f}\n{'='*40}\n")
                if eval_reward > best_mean_reward:
                    best_mean_reward = eval_reward
                    self.save_checkpoint(step, eval_reward)

            # Periodic TMAE update
            if step % cfg.causal_update_freq == 0 and len(buffer) >= 1000:
                sample = buffer.sample(min(1000, len(buffer)))
                agent.tmae.compute_importance(sample[0], sample[1], sample[2])
                print(f"  TMAE weights updated: {agent.tmae.weights}")

    def evaluate(self, n_episodes: int = 10) -> float:
        """Run evaluation episodes without exploration."""
        rewards = []
        for _ in range(n_episodes):
            state = self.env.reset()
            self.agent.reset()
            episode_reward = 0.0
            done = False
            while not done:
                actions = self.agent.select_actions(state, eps=0.0)
                state, reward, done = self.env.step(actions)
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)

    def save_checkpoint(self, step: int, reward: float):
        path = f"checkpoint_step_{step}_reward_{reward:.1f}.pt"
        torch.save({
            "step": step,
            "qmix_state": self.agent.qmix.state_dict(),
            "diff_policies": [dp.state_dict() for dp in self.agent.diffusion_policies],
            "hygma_state": self.agent.hygma.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        data = torch.load(path, map_location=self.agent.device)
        self.agent.qmix.load_state_dict(data["qmix_state"])
        for i, sd in enumerate(data["diff_policies"]):
            self.agent.diffusion_policies[i].load_state_dict(sd)
        self.agent.hygma.load_state_dict(data["hygma_state"])
        print(f"Checkpoint loaded: {path}")
