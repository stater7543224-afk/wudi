import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from src.config import Config
from src.trainer import Trainer


def main():
    cfg = Config()
    trainer = Trainer(cfg)

    # Load latest checkpoint
    checkpoints = [f for f in os.listdir(".") if f.startswith("checkpoint_step_")]
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        trainer.load_checkpoint(latest)
        print(f"Loaded {latest}")

    eval_reward = trainer.evaluate(n_episodes=20)
    print(f"\nFinal evaluation reward: {eval_reward:.2f}")

    # Render one episode
    print("\n--- Rendered Episode ---")
    state = trainer.env.reset()
    total_reward = 0.0
    for t in range(cfg.episode_len):
        actions = trainer.agent.select_actions(state, eps=0.0)
        state, reward, done = trainer.env.step(actions)
        total_reward += reward
        print(f"t={t:2d}  R={reward:6.2f}  cum={total_reward:6.2f}  done={done}")
        if done:
            break
    print(f"Episode total: {total_reward:.2f}")


if __name__ == "__main__":
    main()
