"""Seeding and result I/O shared by the training scripts."""

import os
import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_run(path, episode_returns, mean_q_values=None):
    """Save one run as a CSV with one row per episode."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"episode": np.arange(len(episode_returns)), "return": episode_returns}
    if mean_q_values is not None:
        data["mean_q"] = mean_q_values
    pd.DataFrame(data).to_csv(path, index=False)
