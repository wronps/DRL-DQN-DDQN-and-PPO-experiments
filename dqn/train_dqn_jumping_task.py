"""Double DQN with 40-step returns on three configurations of the Jumping Task.

Each run is saved to results/jumping_task/config_<c>_run_<r>.csv (per-episode return)
and one learning-curve figure per configuration is written to figures/.
An episode is a success when the agent reaches the right edge of the screen, which
pays a +100 bonus, so returns above 100 mark successful episodes.

    uv run python dqn/train_dqn_jumping_task.py                       # 3 configs x 3 runs x 5000 episodes
    uv run python dqn/train_dqn_jumping_task.py --configs 2 --runs 1 --num-training-episodes 200
    uv run python dqn/train_dqn_jumping_task.py --plot-only           # redraw figures from saved results
"""

import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jumping_task.envs import JumpTaskEnv

import double_dqn
import replay_buffer
from agent_environment import agent_environment_episode_loop
from epsilon_greedy_explorers import LinearDecayEpsilonGreedyExploration
from networks import JumpingTaskQNetwork
from plotting import FIGURES_DIR
from utils import save_run, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_STEP = 40
DISCOUNT = 1.0
BUFFER_SIZE = 5000
TARGET_UPDATE_INTERVAL = 100
START_EPSILON = 0.05
END_EPSILON = 1e-6
EPSILON_DECAY_STEPS = 100_000
ADAM_EPS = 1e-4


def get_env(config_num, render=False):
    if config_num == 1:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 25, 30],
                          agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                          obstacle_position=0, obstacle_size=(9, 10),
                          rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                          max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif config_num == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                          agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                          obstacle_position=0, obstacle_size=(11, 17),
                          rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                          max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                          jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                          agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                          obstacle_position=0, obstacle_size=(9, 10),
                          rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                          max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    return env


def input_preprocessor(x):
    """Zero the one-pixel frame border and flatten the 60x60 image to a 3600-d vector."""
    x[:, 0] = 0
    x[:, -1] = 0
    x[0, :] = 0
    x[-1, :] = 0
    return x.reshape(-1)


def run_jumping_task(config, run, num_episodes, min_replay_size_before_updates, device, render=False):
    set_seed(run)
    env = get_env(config, render)
    num_actions = env.action_space.n

    explorer = LinearDecayEpsilonGreedyExploration(START_EPSILON, END_EPSILON, EPSILON_DECAY_STEPS, num_actions)
    q_network = JumpingTaskQNetwork(num_actions=num_actions).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), eps=ADAM_EPS)
    buffer = replay_buffer.ReplayBuffer(BUFFER_SIZE, discount=DISCOUNT, n_step=N_STEP)
    agent = double_dqn.DoubleDQN(q_network, optimizer, buffer, explorer, DISCOUNT, TARGET_UPDATE_INTERVAL,
                                 gradient_update_frequency=1,
                                 input_preprocessor=input_preprocessor,
                                 min_replay_size_before_updates=min_replay_size_before_updates)

    episode_returns, _ = agent_environment_episode_loop(agent, env, num_episodes)
    return episode_returns


def plot_config(runs_returns, file, window=100):
    """Mean return over runs, smoothed with a centred moving average of +/- ``window`` episodes."""
    mean_curve = np.mean(np.array(runs_returns), axis=0)
    smoothed = np.array([np.mean(mean_curve[max(0, i - window):i + window]) for i in range(len(mean_curve))])
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(smoothed)), smoothed, color="r", label="Double DQN (40-step)")
    plt.title("Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.close()


def produce_plots_for_all_configs(results_dir):
    for config in [1, 2, 3]:
        paths = sorted(glob.glob(os.path.join(results_dir, f"config_{config}_run_*.csv")))
        if paths:
            runs_returns = [pd.read_csv(p)["return"].to_numpy() for p in paths]
            plot_config(runs_returns, os.path.join(FIGURES_DIR, f"jumping_task_config{config}.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3], help="run labels, also used as seeds")
    parser.add_argument("--num-training-episodes", type=int, default=5000)
    parser.add_argument("--min-replay-size-before-updates", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="only redraw the figures from saved results")
    parser.add_argument("--results-dir", default=os.path.join(RESULTS_DIR, "jumping_task"))
    args = parser.parse_args()

    if not args.plot_only:
        for config in args.configs:
            for run in args.runs:
                start = time.time()
                returns = run_jumping_task(config, run, args.num_training_episodes,
                                           args.min_replay_size_before_updates, torch.device(args.device),
                                           args.render)
                save_run(os.path.join(args.results_dir, f"config_{config}_run_{run}.csv"), returns)
                success = np.mean(np.asarray(returns[-500:]) > 100)
                print(f"config {config} run {run}: success rate over the last 500 episodes {success:.1%} "
                      f"({time.time() - start:.0f}s)", flush=True)
    produce_plots_for_all_configs(args.results_dir)


if __name__ == "__main__":
    main()
