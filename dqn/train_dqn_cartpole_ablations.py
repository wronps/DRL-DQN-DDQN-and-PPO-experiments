"""One-factor-at-a-time ablations of 1-step DQN on CartPole-v1.

Studies (all other hyperparameters as in train_dqn_cartpole.py):
  target-update  target-network refresh interval in gradient steps: 1, 10, 100
  buffer-size    replay buffer capacity: 500, 1000, 5000, 25000

Runs are saved to results/cartpole_ablations/<study>_<value>_seed<seed>.csv and the
learning curves (mean over seeds) to figures/.

    uv run python dqn/train_dqn_cartpole_ablations.py                       # both studies, seeds 0-4
    uv run python dqn/train_dqn_cartpole_ablations.py --studies buffer-size --seeds 0 1
"""

import argparse
import os
import time

import numpy as np
import torch

import dqn
from plotting import FIGURES_DIR, plot_many_algs
from train_dqn_cartpole import RESULTS_DIR, run_cartpole
from utils import save_run

STUDIES = {
    "target-update": ("target_update_interval", [1, 10, 100]),
    "buffer-size": ("buffer_size", [500, 1000, 5000, 25000]),
}
PLOT_LABELS = {
    "target-update": ("Target Update Interval", "Target Update Ablation (1-step DQN)"),
    "buffer-size": ("Buffer Size", "Replay Buffer Ablation (1-step DQN)"),
}
COLORS = ["r", "g", "b", "m"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", choices=list(STUDIES), default=list(STUDIES))
    parser.add_argument("--values", nargs="+", type=int, default=None,
                        help="subset of values to run (default: all values of the study)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default=os.path.join(RESULTS_DIR, "cartpole_ablations"))
    args = parser.parse_args()

    for study in args.studies:
        param, values = STUDIES[study]
        values = args.values or values
        returns_by_value = []
        for value in values:
            runs_returns = []
            for seed in args.seeds:
                start = time.time()
                returns, _ = run_cartpole(dqn.DQN, 1, seed, torch.device(args.device),
                                          track_q=False, **{param: value})
                save_run(os.path.join(args.results_dir, f"{study}_{value}_seed{seed}.csv"), returns)
                runs_returns.append(returns)
                print(f"{study}={value} seed={seed}: "
                      f"last-50-episode mean return {np.mean(returns[-50:]):.1f} ({time.time() - start:.0f}s)",
                      flush=True)
            returns_by_value.append(runs_returns)

        label, title = PLOT_LABELS[study]
        plot_many_algs(returns_by_value, [f"{label} {v}" for v in values], COLORS[:len(values)],
                       os.path.join(FIGURES_DIR, f"cartpole_ablation_{study.replace('-', '_')}.png"),
                       title=title)


if __name__ == "__main__":
    main()
