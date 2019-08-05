import pandas as pd


def get_all_dataframe(ds):
    """
    Return X_train and X_test concatenated
    """
    return pd.concat([ds.X_train, ds.X_test], axis=0, sort=False)


def sample(ds, nrows):
    """
    Random sampling
    """
