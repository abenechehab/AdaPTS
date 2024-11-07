import numpy as np
from pathlib import Path
import pandas as pd


def load_rl_data(env_name: str, data_label: str) -> np.ndarray:
    """
    Load the expert data for a given environment and data label.

    Args:
        env_name: The name of the environment.
        data_label: The label of the data to load.

    Returns:
        The expert data for the given environment and data label.
    """
    if env_name == "HalfCheetah":
        n_actions = 6  # number of actions in the HalfCheetah system
        n_observations = 17  # number of observations in the HalfCheetah system
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    root_path = Path("/mnt/vdb/abenechehab/dicl-adapters/src/dicl/data/")
    data_path = root_path / f"D4RL_{env_name}_{data_label}.csv"
    X = pd.read_csv(data_path, index_col=0)
    X = X.values.astype("float")

    # find episodes beginnings. the restart column is equal to 1 at the start of
    # an episode, 0 otherwise.
    restart_index = n_observations + n_actions + 1
    restarts = X[:, restart_index]
    episode_starts = np.where(restarts)[0]

    # sample an episode and extract time series
    episode = np.random.choice(np.arange(len(episode_starts)))
    return X[
        episode_starts[episode] : episode_starts[episode]
        + episode_starts[episode + 1]
        - 1
    ]
