import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px


def compare_train_test_hist(X_train, X_test, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

    ax1.hist(X_train[col], bins=bins)
    ax1.set_title(f"Distribution of {col} on train")

    ax2.hist(X_test[col], bins=bins)
    ax2.set_title(f"Distribution of {col} on test")

    fig.show()


def compare_isfraud_hist(X_train, col, bins):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

    ax1.hist(X_train[X_train["label"] == 0][col], bins=bins)
    ax1.set_title(f"Distribution of non fraud {col}")

    ax2.hist(X_train[X_train["label"] == 1][col], bins=bins)
    ax2.set_title(f"Distribution of fraud {col}")

    fig.show()


def categorical_histogram(X, col, title=None, width=400):
    if title == None:
        title = f"Counts of {col}"
    feature_count = (
        X[col]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={col: "count", "index": col})
    )
    return (
        alt.Chart(feature_count)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:N", axis=alt.Axis(title=col)),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            tooltip=[col, "count"],
        )
        .properties(title=title, width=width)
    )


def compare_cat_hist_train_test(X, X_test, col, width=400):
    return categorical_histogram(
        X, col, f"Counts of train {col}", width
    ) | categorical_histogram(X_test, col, f"Counts of test {col}", width)


def compare_cat_hist_isfraud(X, col, width=800):
    return (
        alt.Chart(
            X.groupby(col)["label"]
            .value_counts(dropna=True)
            .reset_index(0)
            .rename(columns={"label": "count", col: "index"})
            .reset_index()
        )
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("index:N", axis=alt.Axis(title="index")),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            color=alt.Color("label:N"),
            tooltip=["index", "count", "label"],
        )
        .properties(title=f"Layered counts of {col}", width=width)
    )


def categorical_horizontal_histogram(X, col, width=400):
    feature_count = (
        X[col]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={col: "count", "index": col})
    )
    return (
        alt.Chart(feature_count)
        .mark_bar()
        .encode(
            y=alt.Y(f"{col}:N", axis=alt.Axis(title=col)),
            x=alt.X("count:Q", axis=alt.Axis(title="Count")),
            tooltip=[col, "count"],
        )
        .properties(title=f"Counts of {col}", width=width)
    )
