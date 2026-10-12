"""DQN vs. Double DQN with 1-, 5- and 10-step returns on CartPole-v1.

Each (agent, n, seed) run is saved to results/cartpole/<agent>_n<n>_seed<seed>.csv
(per-episode return and mean predicted Q-value); the learning curves (mean over seeds)
are written to figures/.

    uv run python dqn/train_dqn_cartpole.py                          # full grid, seeds 0-4
    uv run python dqn/train_dqn_cartpole.py --agents dqn --n-steps 1 --seeds 0
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

import double_dqn
import dqn
import replay_buffer
from agent_environment import agent_environment_episode_loop
from epsilon_greedy_explorers import LinearDecayEpsilonGreedyExploration
from networks import CartPoleQNetwork
from plotting import FIGURES_DIR, plot_many_algs
from utils import save_run, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

AGENTS = {"dqn": dqn.DQN, "ddqn": double_dqn.DoubleDQN}

# Hyperparameters shared by all CartPole experiments (main comparison and ablations).
LR = 1e-4
OPTIMIZER_EPS = 1e-2
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.001
EPSILON_DECAY_STEPS = 12_500
BUFFER_SIZE = 25_000
DISCOUNT = 0.99
TARGET_UPDATE_INTERVAL = 100
MIN_REPLAY_SIZE_BEFORE_UPDATES = 500
MINIBATCH_SIZE = 128
NUM_TRAINING_EPISODES = 500


def run_cartpole(agent_class, n_step, seed, device,
                 buffer_size=BUFFER_SIZE,
                 target_update_interval=TARGET_UPDATE_INTERVAL,
                 num_episodes=NUM_TRAINING_EPISODES,
                 track_q=True):
    """Train one agent on CartPole-v1 and return (episode_returns, mean_q_values)."""
    set_seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    num_actions = env.action_space.n

    q_network = CartPoleQNetwork(env.observation_space.low.size, num_actions).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=LR, eps=OPTIMIZER_EPS)
    explorer = LinearDecayEpsilonGreedyExploration(INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY_STEPS, num_actions)
    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=DISCOUNT, n_step=n_step)
    agent = agent_class(q_network, optimizer, buffer, explorer, DISCOUNT, target_update_interval,
                        min_replay_size_before_updates=MIN_REPLAY_SIZE_BEFORE_UPDATES,
                        minibatch_size=MINIBATCH_SIZE)

    episode_returns, q_values = agent_environment_episode_loop(agent, env, num_episodes, track_q=track_q)
    env.close()
    return episode_returns, q_values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", choices=list(AGENTS), default=list(AGENTS))
    parser.add_argument("--n-steps", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--num-episodes", type=int, default=NUM_TRAINING_EPISODES)
    parser.add_argument("--device", default="cpu", help="cpu is fastest for these small networks")
    parser.add_argument("--results-dir", default=os.path.join(RESULTS_DIR, "cartpole"))
    args = parser.parse_args()

    returns_by_config, q_by_config = {}, {}
    for n_step in args.n_steps:
        for agent_name in args.agents:
            runs_returns, runs_q = [], []
            for seed in args.seeds:
                start = time.time()
                returns, q_values = run_cartpole(AGENTS[agent_name], n_step, seed, torch.device(args.device),
                                                 num_episodes=args.num_episodes)
                save_run(os.path.join(args.results_dir, f"{agent_name}_n{n_step}_seed{seed}.csv"), returns, q_values)
                runs_returns.append(returns)
                runs_q.append(q_values)
                print(f"{agent_name} n={n_step} seed={seed}: "
                      f"last-50-episode mean return {np.mean(returns[-50:]):.1f} ({time.time() - start:.0f}s)",
                      flush=True)
            returns_by_config[agent_name, n_step] = runs_returns
            q_by_config[agent_name, n_step] = runs_q

    if set(args.agents) == set(AGENTS):
        for n_step in args.n_steps:
            plot_many_algs([returns_by_config[a, n_step] for a in AGENTS], ["DQN", "Double DQN"], ["r", "b"],
                           os.path.join(FIGURES_DIR, f"cartpole_dqn_vs_ddqn_n{n_step}.png"))
            plot_many_algs([q_by_config[a, n_step] for a in AGENTS], ["DQN", "Double DQN"], ["r", "b"],
                           os.path.join(FIGURES_DIR, f"cartpole_qvalues_n{n_step}.png"),
                           ylabel="Q-values", title="Q-values")
    colors = ["r", "b", "green", "m", "c"]
    for agent_name in args.agents:
        label = {"dqn": "DQN", "ddqn": "Double DQN"}[agent_name]
        plot_many_algs([returns_by_config[agent_name, n] for n in args.n_steps],
                       [f"{n}-step {label}" for n in args.n_steps], colors[:len(args.n_steps)],
                       os.path.join(FIGURES_DIR, f"cartpole_nstep_{agent_name}.png"))


if __name__ == "__main__":
    main()
