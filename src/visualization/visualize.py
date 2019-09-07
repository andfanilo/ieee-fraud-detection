from itertools import groupby

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from src.visualization.utils import value_counts, value_counts_byfraud

KDE_LW = 0.5


def hist_train_test(X_train, X_test, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))

    # ax1.hist(X_train[col], bins=bins)
    sns.distplot(
        X_train[col].dropna(),
        bins=bins,
        ax=ax1,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1},
    )
    ax1.set_title(f"Distribution of {col} on train")

    # ax2.hist(X_test[col], bins=bins)
    sns.distplot(
        X_test[col].dropna(),
        bins=bins,
        ax=ax2,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1},
    )
    ax2.set_title(f"Distribution of {col} on test")

    fig.show()


def hist_isfraud(X_train, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))

    # ax1.hist(X_train[X_train["label"] == 0][col], bins=bins)
    sns.distplot(
        X_train[X_train["label"] == 0][col].dropna(),
        bins=bins,
        ax=ax1,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1},
    )
    ax1.set_title(f"Distribution of non fraud {col}")

    # ax2.hist(X_train[X_train["label"] == 1][col], bins=bins, facecolor="orange")
    sns.distplot(
        X_train[X_train["label"] == 1][col].dropna(),
        bins=bins,
        ax=ax2,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1, "color": "orange"},
    )
    ax2.set_title(f"Distribution of fraud {col}")

    fig.show()


def grid_distplot(X, quantitative_cols):
    f = pd.melt(X, value_vars=quantitative_cols)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False, size=4)
    g = g.map(
        sns.distplot, "value", kde_kws={"color": "k", "lw": 0.5}, hist_kws={"alpha": 1}
    )
    return g


def grid_countplot(X, qualitative_cols, keep_n_levels=4):
    """If column has too many levels, keep 4 levels with most values by default
    """

    def limited_countplot(x, **kwargs):
        sns.countplot(
            x,
            order=x.value_counts()
            .iloc[: min(kwargs["keep_n_levels"], x.nunique())]
            .index.astype("str"),
        )

    f = pd.melt(X, value_vars=qualitative_cols)
    f["variable"] = f["variable"].astype(str)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False, size=4)
    g = g.map(limited_countplot, "value", keep_n_levels=keep_n_levels)
    g.set_xticklabels(rotation=45)
    return g


def grouped_countplot_fraud(X, col):
    stacked = (
        pd.crosstab(X[col], X["label"])
        .stack()
        .reset_index()
        .rename(columns={0: "value"})
    )
    sns.barplot(x=col, y="value", hue="label", data=stacked)


def grid_countplot_fraud(X, qualitative_cols):
    f = pd.melt(X, id_vars=["label"], value_vars=qualitative_cols)
    g = sns.catplot("label", col="variable", col_wrap=4, data=f, kind="count")
    return g


def grid_pairplot(X, quantative_cols, jitter=False):
    # https://stackoverflow.com/a/55834340

    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)

    # https://stackoverflow.com/a/21276920 for jitter
    def rand_jitter(arr):
        stdev = 0.01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, **kwargs):
        return plt.scatter(rand_jitter(x), rand_jitter(y), **kwargs)

    g = sns.PairGrid(X[quantative_cols + ["label"]], hue="label", diag_sharey=False)
    g.map_diag(plt.hist)
    g.map_upper(hide_current_axis)
    if jitter:
        g.map_lower(jitter, s=2, alpha=0.1)
    else:
        g.map_lower(plt.scatter, s=2, alpha=0.1)
    return g


def grid_violin_fraud(X, quantative_cols):
    # single violin: ax = sns.violinplot(x="id_01", y="label", data=X, orient='h')
    f = pd.melt(X, id_vars=["label"], value_vars=quantative_cols)
    g = sns.catplot(
        data=f,
        x="value",
        y="label",
        col="variable",
        orient="h",
        col_wrap=4,
        kind="violin",
        sharex=False,
        sharey=False,
        size=4,
    )
    return g


def grid_violin_dataset(X, X_test, quantative_cols):
    train = pd.melt(X, value_vars=quantative_cols)
    test = pd.melt(X_test, value_vars=quantative_cols)
    train["origin"] = "train"
    test["origin"] = "test"
    f = pd.concat([train, test])
    g = sns.catplot(
        data=f,
        x="value",
        y="origin",
        col="variable",
        orient="h",
        col_wrap=4,
        kind="violin",
        sharex=False,
        sharey=False,
        size=4,
    )
    return g


def interactive_hist_isfraud(X, col, bins):
    """
    Kind of plt.hist for Altair
    """
    # I think if one automatically created bin is not completed, you get broadcast error, change bons number then
    # or correct code to take empty bins into account
    df = X[[col, "label"]].copy()
    df[col] = pd.cut(df[col], np.linspace(df[col].min() - 1, df[col].max(), bins))
    return interactive_bar_isfraud(df, col)


def interactive_bar_train_test(X, X_test, col, width=400):
    return (
        interactive_bar(X, col, f"Counts of train {col}", width)
        | interactive_bar(X_test, col, f"Counts of test {col}", width).interactive()
    )


def interactive_bar_isfraud(X, col, width=800):
    feature_count = value_counts_byfraud(X, col)
    feature_count[col] = feature_count[col].astype(str)
    return (
        alt.Chart(feature_count)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(f"{col}:N", axis=alt.Axis(title=col), sort=None),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            color=alt.Color("label:N"),
            tooltip=[col, "count", "label"],
        )
        .properties(title=f"Counts of {col} by fraud", width=width)
    ).interactive()


def interactive_barh_isfraud(X, col, width=800):
    feature_count = value_counts_byfraud(X, col)
    feature_count[col] = feature_count[col].astype(str)
    return (
        alt.Chart(feature_count)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("count:Q", axis=alt.Axis(title="Count")),
            y=alt.Y(f"{col}:N", axis=alt.Axis(title=col), sort=None),
            color=alt.Color("label:N"),
            tooltip=[col, "count", "label"],
        )
        .properties(title=f"Counts of {col} by fraud", width=width)
    ).interactive()


def interactive_bar(X, col, title=None, width=400):
    if title == None:
        title = f"Counts of {col}"
    feature_count = value_counts(X, col)

    feature_count[col] = feature_count[col].astype(str)
    return (
        alt.Chart(feature_count)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:N", axis=alt.Axis(title=col), sort=None),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            tooltip=[col, "count"],
        )
        .properties(title=title, width=width)
    ).interactive()
