"""Environment construction with the observation/reward normalisation used by CleanRL.

Observations and rewards are normalised with running statistics and clipped to
[-10, 10]; episode statistics are recorded before reward normalisation, so the logged
returns are the raw environment returns.
"""

import gymnasium as gym
import numpy as np


def make_env(env_id: str, idx: int, run_name: str, gamma: float):
    def thunk():

        env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk
