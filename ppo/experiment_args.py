"""Hyperparameters and command-line arguments for the PPO experiments.

Default values match CleanRL's ppo_continuous_action.py.
"""

import argparse
import os
from dataclasses import dataclass

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@dataclass
class ExperimentArgs:
    exp_name: str = "ppo_experiment"
    env: str = "Ant-v4"
    total_timesteps: int = 2_000_000
    num_runs: int = 1
    seed: int = 1

    torch_deterministic: bool = True
    cuda: bool = True
    output_dir: str = RESULTS_DIR
    debug: bool = True
    log_interval: int = 1000

    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_experiment")
    parser.add_argument("--env", type=str, default="Ant-v4", choices=["Ant-v4", "Walker2d-v4"])
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_runs", type=int, default=1, help="runs use seeds seed, seed+1, ...")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)

    cli_args = parser.parse_args()
    args = ExperimentArgs(**vars(cli_args))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args
