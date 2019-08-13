import numpy as np
import pandas as pd

from src.visualization.utils import value_counts, value_counts_byfraud

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px


def hist_train_test(X_train, X_test, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))

    ax1.hist(X_train[col], bins=bins)
    ax1.set_title(f"Distribution of {col} on train")

    ax2.hist(X_test[col], bins=bins)
    ax2.set_title(f"Distribution of {col} on test")

    fig.show()


def hist_isfraud(X_train, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))

    ax1.hist(X_train[X_train["label"] == 0][col], bins=bins)
    ax1.set_title(f"Distribution of non fraud {col}")

    ax2.hist(X_train[X_train["label"] == 1][col], bins=bins, facecolor='orange')
    ax2.set_title(f"Distribution of fraud {col}")

    fig.show()


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
    return interactive_bar(X, col, f"Counts of train {col}", width) | interactive_bar(
        X_test, col, f"Counts of test {col}", width
    ).interactive()


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
