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


def reduce_mem_usage(df, verbose=True):
    # Function to reduce the DF size
    # https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def reduce_mem_usage_alt(df):
    """
    Reducing Memory Size of dataset
    
    From https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee
    Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    
    Also https://www.dataquest.io/blog/pandas-big-data/
    """
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    logger.info(f"Memory usage of properties dataframe is : {start_mem_usg} MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.

    for col in df.columns:
        # Print current column type
        logger.debug("******************************")
        logger.debug(f"Column: {col}")
        logger.debug(f"dtype before: {df[col].dtype}")

        if df[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = df[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Reput NaN where we have mn - 1 in column
            df[col] = df[col].replace({mn - 1: np.nan})

        # Print new column type
        logger.debug("dtype after: ", df[col].dtype)
        logger.debug("******************************")

    # Print final result
    logger.info("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    logger.info(f"Memory usage is: {mem_usg} MB")
    logger.info(f"This is, {100 * mem_usg / start_mem_usg} % of the initial size")
    logger.info("_________________")
    logger.info("")
    logger.info("Warning: the following numeric columns have missing values")
    logger.info("_________________")
    logger.info("")
    logger.info(",".join(NAlist))
    logger.info("")

    return df
