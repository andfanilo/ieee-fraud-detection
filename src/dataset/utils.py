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


def safe_downcast(
    col,
    max_loss_limit=0.001,
    avg_loss_limit=0.001,
    na_loss_limit=0,
    n_uniq_loss_limit=0,
    fillna=0,
):
    """
    https://www.kaggle.com/alexeykupershtokh/safe-memory-reduction/notebook
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == "float"
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ["float16", "float32"]

    if na_count <= na_loss_limit:
        try_types = ["int8", "int16", "float16", "int32", "float32"]

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == "int"):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if (
            max_loss <= max_loss_limit
            and avg_loss <= avg_loss_limit
            and na_loss <= na_loss_limit
            and n_uniq_loss <= n_uniq_loss_limit
        ):
            return col_tmp

    # field can't be converted
    return col


def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    """
    So my functons address all of these problems. They allow using really minimal amount of memory and guarantee not losing anything (precision, na values, unique values, etc.). And you can do minification on the fly for new columns: df['a/b'] = sd(df['a']/df['b']).

    Also my sd (stands for safe downcast) function is very flexible. If you consider you can allow to lose 0.1 precision when rounding but wanna save more memory, then no problem, just set sd(col, max_loss_limit=0.1, avg_loss_limit=0.1).
    """
    numerics = [
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)

        # numerics
        if col_type in numerics:
            df[col] = safe_downcast(df[col])

        # strings
        if (col_type == "object") and obj_to_cat:
            df[col] = df[col].astype("category")

        if verbose:
            print(
                f"Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}"
            )
        new_na_count = df[col].isna().sum()
        if na_count != new_na_count:
            print(
                f"Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}"
            )
        new_n_uniq = df[col].nunique(dropna=False)
        if n_uniq != new_n_uniq:
            print(
                f"Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}"
            )

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print(
            "Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)".format(
                start_mem, end_mem, percent
            )
        )
    return df


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2
