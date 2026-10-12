"""Uniform experience replay with on-the-fly n-step returns."""

import collections

import numpy as np


class ReplayBuffer:
    """FIFO replay buffer that stores 1-step transitions in insertion order.

    Because consecutive transitions of an episode sit next to each other, an n-step
    transition can be assembled at sampling time by walking forward from the sampled
    index until n steps have been accumulated or the episode ends.
    """

    def __init__(self, buffer_size, discount=0.99, n_step=1):
        self.buffer = collections.deque([], maxlen=buffer_size)
        self.discount = discount
        self.n_step = n_step

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, terminated, truncated):
        transition = {'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state,
                      'discount': self.discount,
                      'terminated': terminated,
                      'truncated': truncated}
        self.buffer.append(transition)

    def create_multistep_transition(self, index):
        """Build the n-step transition that starts at ``index``.

        Returns the discounted sum of up to ``n_step`` rewards, the state reached after
        them, and the discount gamma^k (k = number of rewards summed) to apply to the
        bootstrap value. Accumulation stops early at episode boundaries and at the
        newest transition in the buffer.
        """
        transition = self.buffer[index].copy()
        gamma = self.discount
        accumulated_discount = gamma
        state = transition['state']
        action = transition['action']
        reward = transition['reward']
        next_state = transition['next_state']
        terminated = transition['terminated']
        truncated = transition['truncated']

        for i in range(1, self.n_step):
            if terminated or truncated or index + i >= len(self.buffer):
                break
            next_transition = self.buffer[index + i].copy()
            next_state = next_transition['next_state']
            later_reward = next_transition['reward']
            terminated = next_transition['terminated']
            truncated = next_transition['truncated']
            reward += accumulated_discount * later_reward
            accumulated_discount *= gamma

        return {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'discount': accumulated_discount,
            'terminated': terminated,
            'truncated': truncated
        }

    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch
