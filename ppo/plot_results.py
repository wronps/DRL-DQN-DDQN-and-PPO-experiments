"""Redraw the PPO figures and print the final-return summary from the saved results.

    uv run python ppo/plot_results.py
"""

import os

import numpy as np

from ablation_entropy_walker2d import load_ablation_runs, plot_ablation
from experiment_args import RESULTS_DIR
from train_ppo import FIGURES_DIR, load_runs, plot_multi_seed_results

# Two-sided 95% Student-t critical values for n - 1 degrees of freedom.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def summarize(env_id, returns, last=50):
    finals = np.array([np.mean(r[-last:]) for r in returns])
    n = len(finals)
    half_width = T_975[n - 1] * finals.std(ddof=1) / np.sqrt(n)
    per_seed = ", ".join(f"{f:.0f}" for f in finals)
    print(f"{env_id}: {n} seeds, mean return over the last {last} episodes = "
          f"{finals.mean():.0f} +/- {half_width:.0f} (95% CI); per seed: {per_seed}")


def main():
    for env_id in ["Ant-v4", "Walker2d-v4"]:
        returns, steps = load_runs(os.path.join(RESULTS_DIR, "main"), env_id)
        plot_multi_seed_results(returns, steps, env_id, os.path.join(FIGURES_DIR, f"ppo_{env_id.lower()}.png"))
        summarize(env_id, returns)

    returns_curves, entropy_curves = load_ablation_runs(os.path.join(RESULTS_DIR, "entropy_ablation"))
    plot_ablation(returns_curves, entropy_curves)
    for (coef, _, rets), (_, _, ents) in zip(returns_curves, entropy_curves):
        print(f"Walker2d-v4 ent_coef={coef}: last-50-episode return {np.mean(rets[-50:]):.0f}, "
              f"policy entropy {ents[0]:.1f} -> {ents[-1]:.1f}")


if __name__ == "__main__":
    main()
