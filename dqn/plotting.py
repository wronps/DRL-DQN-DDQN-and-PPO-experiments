"""Learning-curve plots: per-episode mean over runs."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Episodic Returns"):
    """Plot the per-episode mean over runs for several configurations.

    Args:
        lists: one entry per configuration, each a list of runs (list of per-episode values).
        labels, colors: legend label and line colour per configuration.
        file: output path.
    """
    plt.figure(figsize=(10, 6))
    for runs, label, color in zip(lists, labels, colors):
        mean_curve = np.mean(np.array(runs), axis=0)
        plt.plot(range(len(mean_curve)), mean_curve, color=color, label=label)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.close()
