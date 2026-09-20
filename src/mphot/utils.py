"""Small helpers shared across the package."""

import numpy as np
import pandas as pd
from IPython.display import clear_output


def interpolate_dfs(index: list, *data: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates multiple pandas DataFrames based on a given index.

    Args:
        index (list): A list of index values to interpolate over.
        data (pd.DataFrame): Variable number of pandas DataFrames to be interpolated.

    Returns:
        pd.DataFrame: A single DataFrame with interpolated values for the given index.
    """

    df = pd.DataFrame({"tmp": index}, index=index)
    for dat in data:
        dat = dat[~dat.index.duplicated(keep="first")]
        df = pd.concat([df, dat], axis=1)
    df = df.sort_index()
    df = df.interpolate("index").reindex(index)
    df.drop("tmp", axis=1, inplace=True)

    return df


def gaussian(delta: float, sigma: float) -> float:
    """
    Calculate the value of a Gaussian function.

    This function computes the value of a Gaussian (normal) distribution
    for a given delta and sigma.

    Args:
        delta (float): The difference from the mean (x - mu).
        sigma (float): The standard deviation of the distribution.

    Returns:
        float: The value of the Gaussian function at the given delta.
    """

    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(delta**2) / (2 * sigma**2))


def update_progress(progress: float | int) -> None:
    """
    Updates and displays a progress bar in the console.

    Args:
        progress (float or int): A number between 0 and 1 representing the progress percentage.
                                 If an integer is provided, it will be converted to a float.
                                 Values less than 0 will be treated as 0, and values greater than or equal to 1 will be treated as 1.

    Returns:
        None
    """

    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text)
