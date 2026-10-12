"""Double DQN (van Hasselt et al., 2016)."""

import torch

import dqn


class DoubleDQN(dqn.DQN):
    """DQN whose target decouples action selection from action evaluation.

    The greedy next action is chosen by the online network and evaluated by the
    target network, which reduces the maximisation bias of the standard DQN target.
    """

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        with torch.no_grad():
            next_q_values_current = self.q_network(batched_next_states)
            best_current_actions = torch.argmax(next_q_values_current, dim=1, keepdim=True)
            next_q_values_target = self.target_network(batched_next_states)
            best_next_q_values = next_q_values_target.gather(dim=1, index=best_current_actions).squeeze(1)
            targets = batched_rewards + (1 - batch_terminated) * batched_discounts * best_next_q_values
        return targets
