"""Train PPO on a MuJoCo task for one or more seeds.

Each run is saved to results/main/<env>_seed<seed>.csv (global step and return of every
finished training episode); the multi-seed learning curve is written to figures/.

    uv run python ppo/train_ppo.py --env Ant-v4 --num_runs 6        # seeds 1-6
    uv run python ppo/train_ppo.py --env Walker2d-v4 --num_runs 3   # seeds 1-3
"""

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


def plot_multi_seed_results(all_returns, all_steps, env_id: str, output_file: str, window: int = 10):
    plt.figure(figsize=(10, 6))

    for run_returns, run_steps in zip(all_returns, all_steps):
        if len(run_returns) < max(2, window):
            continue
        returns = np.array(run_returns)
        steps = np.array(run_steps)
        smoothed_returns = np.convolve(returns, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window - 1 :]
        plt.plot(smoothed_steps, smoothed_returns, color="blue", alpha=0.2)

    valid_runs = [(r, s) for r, s in zip(all_returns, all_steps) if len(r) > 1 and len(s) > 1]
    if valid_runs:
        max_step = max(steps[-1] for _, steps in valid_runs)
        common_steps = np.linspace(0, max_step, 1000)
        interp_returns = [np.interp(common_steps, steps, returns) for returns, steps in valid_runs]
        avg_returns = np.mean(interp_returns, axis=0)
        if len(avg_returns) >= max(2, window):
            smoothed_avg_returns = np.convolve(avg_returns, np.ones(window) / window, mode="valid")
            smoothed_common_steps = common_steps[window - 1 :]
            plt.plot(smoothed_common_steps, smoothed_avg_returns, color="red", linewidth=2, label="Average Return")

    plt.title(f"PPO Episodic Returns on {env_id}")
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()



FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save_run(path: str, episode_returns, episode_steps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"global_step": episode_steps, "return": episode_returns}).to_csv(path, index=False)


def load_runs(results_dir: str, env_id: str):
    """Load all saved runs of ``env_id`` as (list of returns, list of steps), ordered by seed."""
    paths = sorted(
        (p for p in os.listdir(results_dir) if p.startswith(f"{env_id}_seed") and p.endswith(".csv")),
        key=lambda p: int(p[len(f"{env_id}_seed"):-len(".csv")]),
    )
    frames = [pd.read_csv(os.path.join(results_dir, p)) for p in paths]
    return [f["return"].to_numpy() for f in frames], [f["global_step"].to_numpy() for f in frames]


def run_single_experiment(args, run_seed: int):
    run_name = f"{args.env}__{args.exp_name}__{run_seed}__{int(time.time())}"

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, i, run_name, args.gamma) for i in range(args.num_envs)]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None
    assert action_shape is not None

    agent = ActorCritic(obs_shape, action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    buffer = RolloutBuffer(args.num_steps, args.num_envs, obs_shape, action_shape, device)
    trainer = PPO(agent, optimizer, buffer, args, obs_shape, action_shape)

    global_step = 0
    run_episode_returns = []
    run_episode_steps = []
    next_obs_np, _ = envs.reset(seed=run_seed)
    next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    iterator = tqdm(range(1, args.num_iterations + 1), desc="Training Iterations", leave=False, position=1)
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
        trainer.update(advantages, returns)

        last_episode_return = run_episode_returns[-1] if run_episode_returns else 0.0
        avg_50_return = float(np.mean(run_episode_returns[-50:])) if run_episode_returns else 0.0
        iterator.set_postfix(
            {
                "step": global_step,
                "last_ret": f"{last_episode_return:.2f}",
                "avg50": f"{avg_50_return:.2f}",
            }
        )

        if args.debug and iteration % max(1, args.log_interval) == 0:
            print(
                f"Iter {iteration}/{args.num_iterations} | "
                f"Step {global_step} | "
                f"Last Episode Return: {last_episode_return:.2f} | "
                f"Avg50: {avg_50_return:.2f} | "
            )

    envs.close()
    return run_episode_returns, run_episode_steps


if __name__ == "__main__":
    args = parse_args()
    all_returns = []
    all_steps = []

    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        print(f"Starting run {run_idx + 1}/{args.num_runs} with seed={run_seed}")
        run_returns, run_steps = run_single_experiment(args, run_seed)
        all_returns.append(run_returns)
        all_steps.append(run_steps)
        save_run(os.path.join(args.output_dir, "main", f"{args.env}_seed{run_seed}.csv"), run_returns, run_steps)

    plot_file = os.path.join(FIGURES_DIR, f"ppo_{args.env.lower()}.png")
    plot_multi_seed_results(all_returns, all_steps, args.env, plot_file)
    print(f"Saved plot to {plot_file}")

