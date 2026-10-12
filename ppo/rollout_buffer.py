"""Fixed-size on-policy storage for one PPO rollout (num_steps x num_envs)."""

import torch


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

    def flatten(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        b_obs = self.obs.reshape((-1,) + obs_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + action_shape)
        b_rewards = self.rewards.reshape(-1)
        b_dones = self.dones.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_rewards, b_dones, b_values
