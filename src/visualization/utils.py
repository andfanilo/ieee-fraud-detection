import numpy as np


def value_counts(X, col):
    return (
        X[col]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
        .rename(columns={col: "count", "index": col})
    )


def value_counts_byfraud(X, col):
    return (
        X.groupby(col)["label"]
        .value_counts(dropna=False)
        .reset_index(0)
        .rename(columns={"label": "count"})
        .reset_index()
    )


# https://stackoverflow.com/a/21276920 for jitter
def rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev
