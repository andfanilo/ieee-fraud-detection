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


def sample(ds, nrows):
    """
    Random sampling
    """
