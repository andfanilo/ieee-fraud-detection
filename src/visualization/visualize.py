from itertools import combinations, groupby

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from src.visualization.utils import (
    heatmap,
    rand_jitter,
    value_counts,
    value_counts_byfraud,
)

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

    # ax1.hist(X_train[X_train["isFraud"] == 0][col], bins=bins)
    sns.distplot(
        X_train[X_train["isFraud"] == 0][col].dropna(),
        bins=bins,
        ax=ax1,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1},
    )
    ax1.set_title(f"Distribution of non fraud {col}")

    # ax2.hist(X_train[X_train["isFraud"] == 1][col], bins=bins, facecolor="orange")
    sns.distplot(
        X_train[X_train["isFraud"] == 1][col].dropna(),
        bins=bins,
        ax=ax2,
        kde_kws={"color": "k", "lw": KDE_LW},
        hist_kws={"alpha": 1, "color": "orange"},
    )
    ax2.set_title(f"Distribution of fraud {col}")

    fig.show()


def plot_correlation_matrix(df, cols):
    fig, ax = plt.subplots(figsize=(16, 12))

    # Compute the correlation matrix
    corr = df[cols].corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.show()


def corrplot(df, cols, size_scale=500, marker="s"):
    """Better looking correlation heatmap from https://github.com/drazenz/heatmap/blob/master/heatmap.py
    """
    plt.figure(figsize=(10, 10))
    data = df[cols].corr()
    corr = pd.melt(data.reset_index(), id_vars="index")
    corr.columns = ["x", "y", "value"]
    heatmap(
        corr["x"],
        corr["y"],
        color=corr["value"],
        color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr["value"].abs(),
        size_range=[0, 1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
    )
    plt.show()


def plot_jitter(x, y, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), **kwargs)


def plot_joint(X, col1, col2, jitter=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(col1, fontsize=10)
    ax.set_ylabel(col2, fontsize=10)
    ax.set_title(f"Joint {col1} - {col2}", fontsize=15)
    targets = [0, 1]
    colors = ["g", "r"]
    for target, color in zip(targets, colors):
        indicesToKeep = X["isFraud"] == target
        if jitter:
            plot_jitter(
                X.loc[indicesToKeep, col1], X.loc[indicesToKeep, col2], c=color, s=15
            )
        else:
            ax.scatter(
                X.loc[indicesToKeep, col1], X.loc[indicesToKeep, col2], c=color, s=15
            )
    ax.legend(targets)
    return fig


def save_pairplots(X, cols, folder, jitter=False):
    all_combinations = list(combinations(cols, 2))
    for col1, col2 in all_combinations:
        plot_joint(X, col1, col2, jitter).savefig(folder + f"{col1}-{col2}.png")
        plt.close()


def make_ts_heatmap(source_df, querystr, display_fraud=False, verbose=False):
    width = 480
    height = 183
    img_size = width * height
    seconds_per_pixel = 180

    df = source_df.query(querystr, engine="python")  # filter rows given query string
    if verbose:
        print(querystr, df.shape[0], "transactions")
    ts = (
        df.TransactionDT // seconds_per_pixel
    ).values  # store for each transaction where it should add 1 to image
    c = np.zeros(img_size, dtype=int)
    if display_fraud:
        np.add.at(c, ts, df.isFraud)
    else:
        np.add.at(c, ts, 1)
    return c.reshape((height, width))


def prepare_ts_heatmap_plot(source_df, querystr, display_fraud=False):
    width = 480
    height = 183
    day_marker = 10

    xlabels = [f"{h}am" for h in range(12)] + [f"{h}pm" for h in range(12)]
    xlabels[12] = "12pm"
    ylabels = [f"day{i}" for i in range(0, height, day_marker)]
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams["image.cmap"] = "afmhot"

    fig, ax = plt.subplots(figsize=(18, 6))
    p = make_ts_heatmap(source_df, querystr, display_fraud=display_fraud, verbose=True)
    c = ax.pcolormesh(p)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, width, 20), False)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(0, height, day_marker))
    ax.set_yticklabels(ylabels)
    ax.set_title(querystr)
    cbar = fig.colorbar(c, ax=ax)
    return fig


def show_ts_heatmap(source_df, querystr, display_fraud=False):
    fig = prepare_ts_heatmap_plot(source_df, querystr, display_fraud)
    return plt.tight_layout()


def save_ts_heatmap(source_df, querystr, path, display_fraud=False):
    fig = prepare_ts_heatmap_plot(source_df, querystr, display_fraud)
    fig.savefig(path)
    plt.close()


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
        pd.crosstab(X[col], X["isFraud"])
        .stack()
        .reset_index()
        .rename(columns={0: "value"})
    )
    sns.barplot(x=col, y="value", hue="isFraud", data=stacked)


def grid_countplot_fraud(X, qualitative_cols):
    f = pd.melt(X, id_vars=["isFraud"], value_vars=qualitative_cols)
    g = sns.catplot("isFraud", col="variable", col_wrap=4, data=f, kind="count")
    return g


def grid_pairplot(X, quantative_cols, jitter=False):
    # https://stackoverflow.com/a/55834340

    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)

    g = sns.PairGrid(
        X[quantative_cols + ["isFraud"]],
        hue="isFraud",
        hue_order=[0, 1],
        diag_sharey=False,
    )
    g.map_diag(plt.hist)
    g.map_upper(hide_current_axis)
    if jitter:
        g.map_lower(plot_jitter, s=2.5, alpha=0.2)
    else:
        g.map_lower(plt.scatter, s=2.5, alpha=0.2)
    return g


def grid_violin_fraud(X, quantative_cols):
    # single violin: ax = sns.violinplot(x="id_01", y="isFraud", data=X, orient='h')
    f = pd.melt(X, id_vars=["isFraud"], value_vars=quantative_cols)
    g = sns.catplot(
        data=f,
        x="value",
        y="isFraud",
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
    df = X[[col, "isFraud"]].copy()
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
            tooltip=[col, "count", "isFraud"],
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
            tooltip=[col, "count", "isFraud"],
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
