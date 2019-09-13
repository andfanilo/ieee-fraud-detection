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


def convert_category_cols_lgb(ds):
    for col in ds.categorical_cols:
        # we probably don't need to turn everything to type category, should investigate
        # dropping category type : 0.907806 -> 0.916225
        ds.X_train[col] = ds.X_train[col].fillna(-999)  # .astype("category")
        ds.X_test[col] = ds.X_test[col].fillna(-999)  # .astype("category")
