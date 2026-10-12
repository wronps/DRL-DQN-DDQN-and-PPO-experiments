"""Deep Q-Network (DQN) agent with experience replay and a target network."""

import copy

import numpy as np
import torch


def target_network_refresh(q_network):
    """Return a copy of ``q_network`` to be used as the (fixed) target network."""
    return copy.deepcopy(q_network)


class DQN:
    """Deep Q-Network agent (Mnih et al., 2015).

    Every environment step the transition is stored in the replay buffer. Once the
    buffer holds ``min_replay_size_before_updates`` transitions, the online network
    takes one gradient step every ``gradient_update_frequency`` environment steps on a
    uniformly sampled minibatch, and the target network is replaced by a copy of the
    online network every ``gradient_updates_per_target_refresh`` gradient steps.

    Multi-step returns are handled by the replay buffer: each sampled transition
    carries the accumulated n-step reward and the matching bootstrap discount.
    """

    def __init__(self,
                 q_network,
                 optimizer,
                 replay_buffer,
                 explorer,
                 discount,
                 gradient_updates_per_target_refresh,
                 gradient_update_frequency=1,
                 input_preprocessor=lambda x: x,
                 minibatch_size=16,
                 min_replay_size_before_updates=32,
                 reward_phi=lambda reward: reward):
        self.q_network = q_network
        self.optimizer = optimizer
        self.target_network = target_network_refresh(self.q_network)
        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.discount = discount
        self.gradient_updates_per_target_refresh = gradient_updates_per_target_refresh
        self.gradient_update_frequency = gradient_update_frequency
        self.input_preprocessor = input_preprocessor
        self.minibatch_size = minibatch_size
        self.min_replay_size_before_updates = min_replay_size_before_updates
        self.reward_phi = reward_phi

        self.last_obs = None
        self.last_action = None
        self.step_count = 0
        self.update_count = 0
        self.device = next(q_network.parameters()).device

    def act(self, obs) -> int:
        """Select an action for ``obs`` with the exploration policy."""
        self.last_obs = obs
        processed_obs = self.input_preprocessor(obs)
        obs_tensor = torch.from_numpy(processed_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.q_network(obs_tensor)
        action_values = all_q_values[0].cpu().numpy()
        action = self.explorer.select_action(action_values)
        self.last_action = action
        return action

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        """Bootstrapped targets r + gamma^n * max_a Q_target(s', a), cut at terminal states."""
        with torch.no_grad():
            Q = self.target_network(batched_next_states)
            max_Q, _ = torch.max(Q, dim=1)
            targets = batched_rewards + (1 - batch_terminated) * batched_discounts * max_Q
        return targets

    def gradient_update(self):
        """One gradient step on the squared TD error of a sampled minibatch."""
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        device = self.device
        states_list = [self.input_preprocessor(transition['state']) for transition in minibatch]
        actions_list = [transition['action'] for transition in minibatch]
        rewards_list = [transition['reward'] for transition in minibatch]
        next_states_list = [self.input_preprocessor(transition['next_state']) for transition in minibatch]
        discounts_list = [transition['discount'] for transition in minibatch]
        terminated_list = [transition['terminated'] for transition in minibatch]

        states = torch.from_numpy(np.array(states_list)).to(device, dtype=torch.float32)
        actions = torch.from_numpy(np.array(actions_list)).to(device, dtype=torch.int64)
        rewards = torch.from_numpy(np.array(rewards_list)).to(device, dtype=torch.float32)
        next_states = torch.from_numpy(np.array(next_states_list)).to(device, dtype=torch.float32)
        discounts = torch.from_numpy(np.array(discounts_list)).to(device, dtype=torch.float32)
        terminated = torch.from_numpy(np.array(terminated_list)).to(device, dtype=torch.float32)

        self.optimizer.zero_grad()
        all_q_values = self.q_network(states)
        q_values = all_q_values.gather(dim=1, index=actions.unsqueeze(1))
        targets = self.compute_targets(rewards, next_states, discounts, terminated)
        loss = torch.nn.functional.mse_loss(q_values.squeeze(), targets)
        loss.backward()
        self.optimizer.step()

    def process_transition(self, obs, reward: float, terminated: bool, truncated: bool) -> None:
        """Store the transition caused by the last action and update the networks if due."""
        reward = self.reward_phi(reward)
        if self.last_obs is not None and self.last_action is not None:
            self.replay_buffer.append(
                state=self.last_obs,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                terminated=terminated,
                truncated=truncated
            )
        self.step_count += 1
        if len(self.replay_buffer) >= self.min_replay_size_before_updates:
            if self.step_count % self.gradient_update_frequency == 0:
                self.gradient_update()
                self.update_count += 1
                if self.update_count % self.gradient_updates_per_target_refresh == 0:
                    self.target_network = target_network_refresh(self.q_network)
