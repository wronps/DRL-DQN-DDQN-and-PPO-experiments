"""Entropy-coefficient ablation for PPO on Walker2d-v4.

All coefficients start from the same initial network weights and use the same seed, so
the runs differ only in ent_coef. For each coefficient the script saves
results/entropy_ablation/ent_coef_<c>_episodes.csv (global step, episodic return) and
ent_coef_<c>_updates.csv (global step, mean policy entropy of the last minibatch of
each update); the return and entropy curves are written to figures/.

    uv run python ppo/ablation_entropy_walker2d.py --total_timesteps 1000000 --seed 1
"""

import copy
import os
import random
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from env_factory import make_env
from experiment_args import parse_args
from networks import ActorCritic
from ppo_core import PPO
from rollout_buffer import RolloutBuffer


ENTROPY_COEFS = [0.0, 0.001, 0.01, 0.1, 1.0]
TARGET_ENV = "Walker2d-v4"
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def set_all_seeds(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def smooth_series(y_values, x_values, window: int):
    y = np.asarray(y_values, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if len(y) == 0 or len(x) == 0:
        return y, x
    if window <= 1 or len(y) < window:
        return y, x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed_y = np.convolve(y, kernel, mode="valid")
    smoothed_x = x[window - 1 :]
    return smoothed_y, smoothed_x


def plot_ablation_curves(curves_by_coef, ylabel: str, title: str, output_file: str, window: int = 10):
    plt.figure(figsize=(10, 6))
    for coef, x_values, y_values in curves_by_coef:
        if len(y_values) == 0 or len(x_values) == 0:
            continue
        smooth_y, smooth_x = smooth_series(y_values, x_values, window)
        if len(smooth_y) == 1:
            # A single point has no visible line segment, so draw it as a marker.
            plt.plot(smooth_x, smooth_y, marker="o", linestyle="None", markersize=6, label=f"ent_coef={coef}")
        else:
            plt.plot(smooth_x, smooth_y, linewidth=2, label=f"ent_coef={coef}")

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.grid(True)
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()


def save_ablation_run(results_dir: str, ent_coef: float, episode_returns, episode_steps,
                      update_entropies, update_steps) -> None:
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame({"global_step": episode_steps, "return": episode_returns}).to_csv(
        os.path.join(results_dir, f"ent_coef_{ent_coef}_episodes.csv"), index=False)
    pd.DataFrame({"global_step": update_steps, "entropy": update_entropies}).to_csv(
        os.path.join(results_dir, f"ent_coef_{ent_coef}_updates.csv"), index=False)


def load_ablation_runs(results_dir: str):
    """Return (returns_curves, entropy_curves) as lists of (coef, steps, values)."""
    returns_curves, entropy_curves = [], []
    for ent_coef in ENTROPY_COEFS:
        episodes = pd.read_csv(os.path.join(results_dir, f"ent_coef_{ent_coef}_episodes.csv"))
        updates = pd.read_csv(os.path.join(results_dir, f"ent_coef_{ent_coef}_updates.csv"))
        returns_curves.append((ent_coef, episodes["global_step"].to_numpy(), episodes["return"].to_numpy()))
        entropy_curves.append((ent_coef, updates["global_step"].to_numpy(), updates["entropy"].to_numpy()))
    return returns_curves, entropy_curves


def plot_ablation(returns_curves, entropy_curves, window: int = 10) -> None:
    plot_ablation_curves(returns_curves, ylabel="Return", title="PPO Return Ablation on Walker2d-v4",
                         output_file=os.path.join(FIGURES_DIR, "walker2d_return_ablation.png"), window=window)
    plot_ablation_curves(entropy_curves, ylabel="Entropy", title="PPO Entropy Ablation on Walker2d-v4",
                         output_file=os.path.join(FIGURES_DIR, "walker2d_entropy_ablation.png"), window=window)


def build_baseline_state(args, device: torch.device):
    set_all_seeds(args.seed, args.torch_deterministic)
    probe_envs = gym.vector.SyncVectorEnv(
        [make_env(TARGET_ENV, i, f"{TARGET_ENV}__baseline_probe", args.gamma) for i in range(args.num_envs)]
    )

    obs_shape = probe_envs.single_observation_space.shape
    action_shape = probe_envs.single_action_space.shape
    assert obs_shape is not None
    assert action_shape is not None

    baseline_agent = ActorCritic(obs_shape, action_shape).to(device)
    state = copy.deepcopy(baseline_agent.state_dict())
    probe_envs.close()
    return state, obs_shape, action_shape


def run_ablation_once(args, run_seed: int, ent_coef: float, init_state, obs_shape, action_shape, device):
    run_name = f"{TARGET_ENV}__entropy_ablation__coef_{ent_coef}__seed_{run_seed}__{int(time.time())}"

    set_all_seeds(run_seed, args.torch_deterministic)

    envs = gym.vector.SyncVectorEnv([make_env(TARGET_ENV, i, run_name, args.gamma) for i in range(args.num_envs)])

    agent = ActorCritic(obs_shape, action_shape).to(device)
    agent.load_state_dict(init_state)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    buffer = RolloutBuffer(args.num_steps, args.num_envs, obs_shape, action_shape, device)

    args.ent_coef = ent_coef
    trainer = PPO(agent, optimizer, buffer, args, obs_shape, action_shape)

    global_step = 0
    run_episode_returns = []
    run_episode_steps = []
    update_entropies = []
    update_steps = []

    next_obs_np, _ = envs.reset(seed=run_seed)
    next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    iterator = tqdm(
        range(1, args.num_iterations + 1),
        desc=f"coef={ent_coef}",
        leave=False,
        position=1,
    )
    for iteration in iterator:
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        next_obs, next_done, global_step, episode_returns, episode_steps = trainer.collect_rollout(
            envs, next_obs, next_done, global_step
        )
        run_episode_returns.extend(episode_returns)
        run_episode_steps.extend(episode_steps)

        advantages, returns = trainer.compute_advantages_and_returns(next_obs, next_done)
        metrics = trainer.update(advantages, returns)

        update_entropies.append(float(metrics["entropy"]))
        update_steps.append(int(global_step))

        last_episode_return = run_episode_returns[-1] if run_episode_returns else 0.0
        avg_50_return = float(np.mean(run_episode_returns[-50:])) if run_episode_returns else 0.0
        iterator.set_postfix(
            {
                "step": global_step,
                "last_ret": f"{last_episode_return:.2f}",
                "avg50": f"{avg_50_return:.2f}",
                "entropy": f"{metrics['entropy']:.4f}",
            }
        )

        if args.debug and iteration % max(1, args.log_interval) == 0:
            print(
                f"coef={ent_coef} | Iter {iteration}/{args.num_iterations} | "
                f"Step {global_step} | LastRet {last_episode_return:.2f} | "
                f"Avg50 {avg_50_return:.2f} | Entropy {metrics['entropy']:.4f}"
            )

    envs.close()
    return run_episode_returns, run_episode_steps, update_entropies, update_steps


def main():
    args = parse_args()

    if args.env != TARGET_ENV:
        print(f"Overriding env from {args.env} to {TARGET_ENV} for entropy ablation.")
    args.env = TARGET_ENV

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    init_state, obs_shape, action_shape = build_baseline_state(args, device)

    returns_curves = []
    entropy_curves = []

    for ent_coef in ENTROPY_COEFS:
        print(f"\nStarting entropy ablation run: ent_coef={ent_coef}, seed={args.seed}")
        episode_returns, episode_steps, update_entropies, update_steps = run_ablation_once(
            args=args,
            run_seed=args.seed,
            ent_coef=ent_coef,
            init_state=init_state,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
        )

        save_ablation_run(os.path.join(args.output_dir, "entropy_ablation"), ent_coef,
                          episode_returns, episode_steps, update_entropies, update_steps)

        returns_curves.append((ent_coef, episode_steps, episode_returns))
        entropy_curves.append((ent_coef, update_steps, update_entropies))

        final_return = float(episode_returns[-1]) if episode_returns else 0.0
        avg_50_return = float(np.mean(episode_returns[-50:])) if episode_returns else 0.0
        final_entropy = float(update_entropies[-1]) if update_entropies else 0.0
        avg_entropy = float(np.mean(update_entropies)) if update_entropies else 0.0
        print(
            f"Summary ent_coef={ent_coef}: final_return={final_return:.2f}, "
            f"avg50_return={avg_50_return:.2f}, final_entropy={final_entropy:.4f}, "
            f"mean_entropy={avg_entropy:.4f}"
        )

    plot_ablation(returns_curves, entropy_curves)
    print(f"Saved return and entropy plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
