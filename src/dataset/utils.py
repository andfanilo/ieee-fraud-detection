import numpy as np
import pandas as pd


def df_empty(columns, dtypes, index=None):
    """
    Create Empty Dataframe in Pandas specifying column types
    https://stackoverflow.com/a/48374031
    """
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def get_all_dataframe(ds):
    """
    Return X_train and X_test concatenated
    """
    return pd.concat([ds.X_train, ds.X_test], axis=0, sort=False)


def reduce_mem_usage(df, deep=True, verbose=True, categories=False):
    # Function to reduce the DF size
    # https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
    # Don't use float16 : https://www.kaggle.com/c/ieee-fraud-detection/discussion/107653#latest-619384
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if categories and col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(
            f"Memory usage decreased from"
            f" {start_mem:.2f}MB to {end_mem:.2f}MB"
            f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)"
        )

    return df


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2
