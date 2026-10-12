"""PPO with a clipped surrogate objective, GAE and a clipped value loss.

The update rule and its defaults follow CleanRL's ppo_continuous_action.py
(https://github.com/vwxyzjn/cleanrl); the code is split into rollout collection,
advantage estimation and the minibatch update.
"""

import numpy as np
import torch
import torch.nn as nn

from experiment_args import ExperimentArgs
from networks import ActorCritic
from rollout_buffer import RolloutBuffer


class PPO:
    def __init__(
        self,
        agent: ActorCritic,
        optimizer: torch.optim.Optimizer,
        buffer: RolloutBuffer,
        args: ExperimentArgs,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.buffer = buffer
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def collect_rollout(self, envs, next_obs: torch.Tensor, next_done: torch.Tensor, global_step: int):
        episode_returns = []
        episode_steps = []
        for step in range(self.args.num_steps):
            global_step += self.args.num_envs
            self.buffer.obs[step] = next_obs
            self.buffer.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.buffer.values[step] = value.flatten()

            self.buffer.actions[step] = action
            self.buffer.logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            self.buffer.rewards[step] = torch.tensor(reward, device=self.buffer.device).view(-1)
            next_obs = torch.tensor(next_obs_np, device=self.buffer.device, dtype=torch.float32)
            next_done = torch.tensor(next_done_np, device=self.buffer.device, dtype=torch.float32)

            if "episode" in infos:
                episode_info = infos["episode"]
                returns = np.asarray(episode_info["r"])
                lengths = np.asarray(episode_info["l"])
                return_mask = np.asarray(episode_info.get("_r", np.ones_like(returns, dtype=bool))).astype(bool)
                length_mask = np.asarray(episode_info.get("_l", np.ones_like(lengths, dtype=bool))).astype(bool)
                valid_mask = return_mask & length_mask

                for ret, ep_len, is_valid in zip(returns, lengths, valid_mask):
                    if not is_valid:
                        continue
                    ret_value = float(ret)
                    print(f"global_step={global_step}, episodic_return={ret_value}")
                    episode_returns.append(ret_value)
                    episode_steps.append(int(global_step))

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ret_value = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                        print(f"global_step={global_step}, episodic_return={ret_value}")
                        episode_returns.append(ret_value)
                        episode_steps.append(int(global_step))

        return next_obs, next_done, global_step, episode_returns, episode_steps

    def compute_advantages_and_returns(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.buffer.rewards, device=self.buffer.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.buffer.dones[t + 1]
                    nextvalues = self.buffer.values[t + 1]
                delta = self.buffer.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
                lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + self.buffer.values

        return advantages, returns

    def update(self, advantages: torch.Tensor, returns: torch.Tensor):
        b_obs, b_logprobs, b_actions, _, _, b_values = self.buffer.flatten(self.obs_shape, self.action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=self.buffer.device)
        approx_kl = torch.tensor(0.0, device=self.buffer.device)
        pg_loss = torch.tensor(0.0, device=self.buffer.device)
        v_loss = torch.tensor(0.0, device=self.buffer.device)
        entropy_loss = torch.tensor(0.0, device=self.buffer.device)

        for _ in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": explained_var,
        }
