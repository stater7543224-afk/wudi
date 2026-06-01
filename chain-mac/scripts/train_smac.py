import argparse, sys, os, numpy as np, torch, time
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config_smac import make_smac_config
from src.env.smac_env import SMACEnv
from src.agent import CHAIN_MAC
from src.buffer import ReplayBuffer


class Trainer:
    def __init__(self, cfg, num_envs=16):
        self.cfg = cfg
        self.num_envs = num_envs
        self.envs = [SMACEnv(map_name=cfg.map_name) for _ in range(num_envs)]
        self.agent = CHAIN_MAC(cfg)
        print("device:", self.agent.device)

        self.buf = ReplayBuffer(
            cfg.buffer_capacity, cfg.n_agents, cfg.obs_dim,
            state_dim=cfg.state_dim)

        self.ep_rew = [deque(maxlen=100) for _ in range(num_envs)]
        self.ep_win = [deque(maxlen=100) for _ in range(num_envs)]

    def train(self):
        cfg, agent, buf = self.cfg, self.agent, self.buf
        n = self.num_envs
        best = -9e9

        obs = [e.reset() for e in self.envs]
        for _ in range(n):
            agent.reset()
        ep_r = [0.0] * n
        next_log = 0
        t0 = time.time()
        total = cfg.total_train_steps
        step = 0

        while step < total:
            eps = max(0.05, 1.0 - step / (total * 0.5))
            avail = [e.get_avail_actions() for e in self.envs]

            # select actions for all envs
            acts = []
            for i in range(n):
                a = agent.select_actions(obs[i], eps=eps, avail_mask=avail[i])
                for j in range(len(a)):
                    if not avail[i][j, a[j]]:
                        v = np.where(avail[i][j])[0]
                        a[j] = np.random.choice(v) if len(v) > 0 else 0
                acts.append(a)

            for i in range(n):
                gs = self.envs[i].get_global_state()
                no, r, done, info = self.envs[i].step(acts[i])
                ngs = self.envs[i].get_global_state()
                buf.push(obs[i], acts[i], r, no, done,
                         global_state=gs, next_global_state=ngs)
                ep_r[i] += r
                obs[i] = no
                step += 1

                if done:
                    self.ep_rew[i].append(ep_r[i])
                    won = info.get("battle_won", False) or r > 10
                    self.ep_win[i].append(1.0 if won else 0.0)
                    obs[i] = self.envs[i].reset()
                    agent.reset()
                    ep_r[i] = 0.0

                if step >= total:
                    break

            # training
            if step % cfg.learn_freq == 0 and len(buf) >= cfg.batch_size:
                b = buf.sample(cfg.batch_size)
                l = agent.update(b[0], b[1], b[2], b[3], b[4],
                                 global_states=b[5], next_global_states=b[6])

                if step >= next_log:
                    next_log = step + cfg.log_freq
                    ar, aw = [], []
                    for d in self.ep_rew:
                        if d: ar.extend(d)
                    for d in self.ep_win:
                        if d: aw.extend(d)
                    mr = np.mean(ar) if ar else 0.0
                    mw = np.mean(aw) if aw else 0.0
                    sps = step / (time.time() - t0 + 1e-8)
                    eta = (total - step) / sps / 60 if sps > 0 else 0
                    print("Step %d | eps %.3f | R %.2f | win %d%%%% | Q %.4f | %d sps | ETA %.0fm" % (
                        step, eps, mr, int(mw*100), l["qmix_loss"], int(sps), eta))

            if step % cfg.eval_freq == 0 and step > 0:
                er, ew = self.evaluate(5)
                print("** Eval %d: R=%.2f win=%d%%%% **" % (step, er, int(ew*100)))
                if er > best:
                    best = er
                    torch.save({
                        "step": step,
                        "qmix": agent.qmix.state_dict(),
                        "diff": [dp.state_dict() for dp in agent.diffusion_policies],
                        "hygma": agent.hygma.state_dict(),
                    }, "ckpt_%d.pt" % step)
                    print("  -> saved")

    def evaluate(self, n=10):
        rw = []
        for _ in range(n):
            s = self.envs[0].reset()
            self.agent.reset()
            ep = 0.0
            done = False
            while not done:
                a = self.agent.select_actions(s, eps=0.0, avail_mask=self.envs[0].get_avail_actions())
                s, r, done, _ = self.envs[0].step(a)
                ep += r
            rw.append(ep)
        return float(np.mean(rw)), sum(1 for x in rw if x > 0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="3m")
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--q-boost", type=float, default=0.01)
    p.add_argument("--num-envs", type=int, default=16)
    args = p.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = make_smac_config(args.map, args.groups, args.steps,
                           batch_size=args.batch, lr=args.lr,
                           q_boost_lambda=args.q_boost)
    cfg.map_name = args.map

    print("CHAIN-MAC | SMAC %s | %d envs | %d agents | %d obs | %d act" % (
        args.map, args.num_envs, cfg.n_agents, cfg.obs_dim, cfg.action_dim))
    print("steps=%d batch=%d lr=%g q_boost=%g" % (
        cfg.total_train_steps, cfg.batch_size, cfg.lr, cfg.q_boost_lambda))

    Trainer(cfg, args.num_envs).train()


if __name__ == "__main__":
    main()
