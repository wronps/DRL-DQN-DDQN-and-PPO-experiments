# DQN, Double DQN and PPO experiments

PyTorch implementations of value-based (DQN, Double DQN with n-step returns) and policy-gradient (PPO) deep RL agents, with the experiments and ablations run on them. The work was done for a graduate course in deep reinforcement learning.

| Folder | Environments | Studied |
|---|---|---|
| [`dqn/`](dqn/) | CartPole-v1; Jumping Task (60×60 pixels) | DQN vs. Double DQN, 1/5/10-step returns, Q-value estimates, target-update interval, replay-buffer size |
| [`ppo/`](ppo/) | Ant-v4, Walker2d-v4 (MuJoCo) | PPO over several seeds; weight of the entropy bonus |

Each folder has its own README with the setup, all figures and a discussion of the results.

## Results

- **PPO, 2M steps.** Mean training return over the last 50 episodes: 3238 ± 346 on Walker2d-v4 (3 seeds) and 2714 ± 752 on Ant-v4 (6 seeds; per seed 1941 to 3723). ± is a 95% t-interval over seeds.
- **PPO entropy ablation** (Walker2d-v4, one seed per value). With an entropy weight of 0.1 or 1.0 the learned log standard deviation grows without bound and the agent never learns to walk; weights of 0, 0.001 and 0.01 all learn to walk.
- **Double DQN on the Jumping Task** (40-step returns, 3 runs per configuration). Success in 99.7% and 99.8% of the last 500 training episodes on configurations 2 and 3, and 83.1% on configuration 1.
- **CartPole.** With 5- and 10-step returns both DQN and Double DQN start improving about 35–55 episodes earlier than with 1-step returns. With 1-step returns the Q-estimates of both algorithms temporarily exceed the largest possible value; with 5- and 10-step returns they do not.

<p align="center">
  <img src="ppo/figures/ppo_walker2d-v4.png" width="49%" alt="PPO on Walker2d-v4, 3 seeds">
  <img src="ppo/figures/ppo_ant-v4.png" width="49%" alt="PPO on Ant-v4, 6 seeds">
</p>
<p align="center"><sub>PPO training returns. Light lines: individual seeds; red: mean over seeds.</sub></p>

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python dqn/train_dqn_cartpole.py
uv run python ppo/train_ppo.py --env Walker2d-v4 --num_runs 3
```

Gymnasium (1.2.3) and MuJoCo (3.5.0) are pinned to the versions used for the results. The data behind the figures is in `dqn/results/` and `ppo/results/`; these commands redraw the figures from it without training:

```bash
uv run python ppo/plot_results.py
uv run python dqn/train_dqn_jumping_task.py --plot-only
```

The CartPole runs kept only their figures, not per-run data.

## Acknowledgements

- The DQN part started from course starter code: skeletons of the agent, replay buffer and episode loop, the CartPole training script with its hyperparameters and plotting helpers, and the Jumping Task configurations.
- The PPO implementation follows CleanRL's [`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl).
- Jumping Task environment: [prabhatnagarajan/jumping-task](https://github.com/prabhatnagarajan/jumping-task), a Gymnasium version of the task from Tachet des Combes et al. (2018).
