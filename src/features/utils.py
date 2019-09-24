import numpy as np
import pandas as pd


def calc_smooth_mean(df, by, on, m):
    """
    https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
    """
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(["count", "mean"])
    counts = agg["count"]
    means = agg["mean"]

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


def num_after_point(x):
    """Get number of decimals
    """
    s = str(x)
    if not "." in s:
        return 0
    return len(s) - s.index(".") - 1


def cross_count(X, col1, col2):
    stacked = (
        pd.crosstab(X[col1], X[col2]).stack().reset_index().rename(columns={0: "value"})
    )
    return stacked


def compute_polar_coords(X, col_X, col_Y):
    return (
        np.sqrt(X.eval(f"{col_X}*{col_X} + {col_Y}*{col_Y}")),
        np.arctan((X[col_X] + 1e-9) / (X[col_Y] + 1e-9)),
    )
