"""Epsilon-greedy exploration."""

import numpy as np


def compute_epsilon_greedy_action_probs(q_vals, epsilon):
    """Epsilon-greedy action probabilities for a 1-D array of Q-values.

    Ties between greedy actions are broken evenly.

    Args:
        q_vals: numpy array of action values.
        epsilon: exploration rate in [0, 1].

    Returns:
        numpy array of action probabilities with the same shape as ``q_vals``.
    """
    assert len(q_vals.shape) == 1
    action_probabilities = np.ones(q_vals.shape[0]) * (epsilon / q_vals.shape[0])
    max_q = np.max(q_vals)
    is_greedy = (max_q == q_vals)
    greedy_num = np.sum(is_greedy)
    action_probabilities[is_greedy] += (1 - epsilon) / greedy_num
    assert action_probabilities.shape == q_vals.shape
    return action_probabilities


class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with epsilon annealed linearly over action selections.

    Args:
        start_epsilon: epsilon at the first action selection.
        end_epsilon: final epsilon, kept after ``decay_steps`` selections.
        decay_steps: number of action selections over which epsilon is annealed.
        num_actions: number of discrete actions.
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, num_actions):
        assert start_epsilon >= end_epsilon >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon = start_epsilon
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.steps = 0

    def select_action(self, action_values) -> int:
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        self.epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)
