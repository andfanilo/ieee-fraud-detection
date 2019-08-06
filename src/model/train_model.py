import time
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

from src.model.utils import eval_auc, group_mean_log_mae

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RepeatedKFold,
    GroupKFold,
    GridSearchCV,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn import metrics


def train_xgb(ds):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        n_jobs=4,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        missing=-999,
    )

    clf.fit(ds.X_train, ds.y_train)
    return clf


def train_lgb(ds):
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        n_jobs=4,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        random_state=314,
    )

    clf.fit(ds.X_train, ds.y_train)
    return clf


def train_model_regression(
    X,
    X_test,
    y,
    params,
    folds=None,
    model_type="lgb",
    eval_metric="mae",
    columns=None,
    plot_feature_importance=False,
    model=None,
    verbose=10000,
    early_stopping_rounds=200,
    n_estimators=50000,
    splits=None,
    n_folds=3,
):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds

    # to set up scoring parameters
    metrics_dict = {
        "mae": {
            "lgb_metric_name": "mae",
            "catboost_metric_name": "MAE",
            "sklearn_scoring_function": metrics.mean_absolute_error,
        },
        "group_mae": {
            "lgb_metric_name": "mae",
            "catboost_metric_name": "MAE",
            "scoring_function": group_mean_log_mae,
        },
        "mse": {
            "lgb_metric_name": "mse",
            "catboost_metric_name": "MSE",
            "sklearn_scoring_function": metrics.mean_squared_error,
        },
    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f"Fold {fold_n + 1} started at {time.ctime()}")
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = (
                X[columns].iloc[train_index],
                X[columns].iloc[valid_index],
            )
            y_train, y_valid = y[train_index], y[valid_index]

        if model_type == "lgb":
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=metrics_dict[eval_metric]["lgb_metric_name"],
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == "xgb":
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns
            )
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns
            )

            watchlist = [(train_data, "train"), (valid_data, "valid_data")]
            model = xgb.train(
                dtrain=train_data,
                num_boost_round=20000,
                evals=watchlist,
                early_stopping_rounds=200,
                verbose_eval=verbose,
                params=params,
            )
            y_pred_valid = model.predict(
                xgb.DMatrix(X_valid, feature_names=X.columns),
                ntree_limit=model.best_ntree_limit,
            )
            y_pred = model.predict(
                xgb.DMatrix(X_test, feature_names=X.columns),
                ntree_limit=model.best_ntree_limit,
            )

        if model_type == "sklearn":
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1)
            score = metrics_dict[eval_metric]["sklearn_scoring_function"](
                y_valid, y_pred_valid
            )
            print(f"Fold {fold_n}. {eval_metric}: {score:.4f}.")
            print("")

            y_pred = model.predict(X_test).reshape(-1)

        if model_type == "cat":
            model = CatBoostRegressor(
                iterations=20000,
                eval_metric=metrics_dict[eval_metric]["catboost_metric_name"],
                **params,
                loss_function=metrics_dict[eval_metric]["catboost_metric_name"],
            )
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                cat_features=[],
                use_best_model=True,
                verbose=False,
            )

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1)
        if eval_metric != "group_mae":
            scores.append(
                metrics_dict[eval_metric]["sklearn_scoring_function"](
                    y_valid, y_pred_valid
                )
            )
        else:
            scores.append(
                metrics_dict[eval_metric]["scoring_function"](
                    y_valid, y_pred_valid, X_valid["type"]
                )
            )

        prediction += y_pred

        if model_type == "lgb" and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )

    prediction /= n_splits
    print(
        "CV mean score: {0:.4f}, std: {1:.4f}.".format(np.mean(scores), np.std(scores))
    )

    result_dict["oof"] = oof
    result_dict["prediction"] = prediction
    result_dict["scores"] = scores

    if model_type == "lgb":
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = (
                feature_importance[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:50]
                .index
            )

            best_features = feature_importance.loc[
                feature_importance.feature.isin(cols)
            ]

            plt.figure(figsize=(16, 12))
            sns.barplot(
                x="importance",
                y="feature",
                data=best_features.sort_values(by="importance", ascending=False),
            )
            plt.title("LGB Features (avg over folds)")

            result_dict["feature_importance"] = feature_importance

    return result_dict


def train_model_classification(
    X,
    X_test,
    y,
    params,
    folds,
    model_type="lgb",
    eval_metric="auc",
    columns=None,
    plot_feature_importance=False,
    model=None,
    verbose=10000,
    early_stopping_rounds=200,
    n_estimators=50000,
    splits=None,
    n_folds=3,
    averaging="usual",
    n_jobs=-1,
):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    From https://www.kaggle.com/artgor/eda-and-models
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {
        "auc": {
            "lgb_metric_name": eval_auc,
            "catboost_metric_name": "AUC",
            "sklearn_scoring_function": metrics.roc_auc_score,
        }
    }

    result_dict = {}
    if averaging == "usual":
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    elif averaging == "rank":
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f"Fold {fold_n + 1} started at {time.ctime()}")
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = (
                X[columns].iloc[train_index],
                X[columns].iloc[valid_index],
            )
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == "lgb":
            model = lgb.LGBMClassifier(
                **params, n_estimators=n_estimators, n_jobs=n_jobs
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=metrics_dict[eval_metric]["lgb_metric_name"],
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[
                :, 1
            ]

        if model_type == "xgb":
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns
            )
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns
            )

            watchlist = [(train_data, "train"), (valid_data, "valid_data")]
            model = xgb.train(
                dtrain=train_data,
                num_boost_round=n_estimators,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose,
                params=params,
            )
            y_pred_valid = model.predict(
                xgb.DMatrix(X_valid, feature_names=X.columns),
                ntree_limit=model.best_ntree_limit,
            )
            y_pred = model.predict(
                xgb.DMatrix(X_test, feature_names=X.columns),
                ntree_limit=model.best_ntree_limit,
            )

        if model_type == "sklearn":
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1)
            score = metrics_dict[eval_metric]["sklearn_scoring_function"](
                y_valid, y_pred_valid
            )
            print(f"Fold {fold_n}. {eval_metric}: {score:.4f}.")
            print("")

            y_pred = model.predict_proba(X_test)

        if model_type == "cat":
            model = CatBoostClassifier(
                iterations=n_estimators,
                eval_metric=metrics_dict[eval_metric]["catboost_metric_name"],
                **params,
                loss_function=metrics_dict[eval_metric]["catboost_metric_name"],
            )
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                cat_features=[],
                use_best_model=True,
                verbose=False,
            )

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if averaging == "usual":

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(
                metrics_dict[eval_metric]["sklearn_scoring_function"](
                    y_valid, y_pred_valid
                )
            )

            prediction += y_pred.reshape(-1, 1)

        elif averaging == "rank":

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(
                metrics_dict[eval_metric]["sklearn_scoring_function"](
                    y_valid, y_pred_valid
                )
            )

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

        if model_type == "lgb" and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )

    prediction /= n_splits

    print(
        "CV mean score: {0:.4f}, std: {1:.4f}.".format(np.mean(scores), np.std(scores))
    )

    result_dict["oof"] = oof
    result_dict["prediction"] = prediction
    result_dict["scores"] = scores

    if model_type == "lgb":
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = (
                feature_importance[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:50]
                .index
            )

            best_features = feature_importance.loc[
                feature_importance.feature.isin(cols)
            ]

            plt.figure(figsize=(16, 12))
            sns.barplot(
                x="importance",
                y="feature",
                data=best_features.sort_values(by="importance", ascending=False),
            )
            plt.title("LGB Features (avg over folds)")

            result_dict["feature_importance"] = feature_importance
            result_dict["top_columns"] = cols

    return result_dict


def train_lgb_folds(ds):
    n_fold = 5
    folds = TimeSeriesSplit(n_splits=n_fold)
    folds = KFold(n_splits=5)

    params = {
        "num_leaves": 256,
        "min_child_samples": 79,
        "objective": "binary",
        "max_depth": 13,
        "learning_rate": 0.03,
        "boosting_type": "gbdt",
        "subsample_freq": 3,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": "auc",
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "colsample_bytree": 0.9,
        #'categorical_feature': cat_cols
    }

    result_dict_lgb = train_model_classification(
        X=ds.X_train,
        X_test=ds.X_test,
        y=ds.y_train,
        params=params,
        folds=folds,
        model_type="lgb",
        eval_metric="auc",
        plot_feature_importance=False,
        verbose=500,
        early_stopping_rounds=200,
        n_estimators=5000,
        averaging="usual",
        n_jobs=-1,
    )

    ds.submission["isFraud"] = result_dict_lgb["prediction"]
    return None


def train_xgb_folds(ds):
    n_fold = 5
    folds = TimeSeriesSplit(n_splits=n_fold)
    folds = KFold(n_splits=5)

    params = {
        "eta": 0.05,
        "max_depth": 9,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "gamma": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "missing": -999,
        "verbosity": 1,
        "nthread": -1,
    }

    result_dict_lgb = train_model_classification(
        X=ds.X_train,
        X_test=ds.X_test,
        y=ds.y_train,
        params=params,
        folds=folds,
        model_type="xgb",
        eval_metric="auc",
        plot_feature_importance=False,
        verbose=500,
        early_stopping_rounds=200,
        n_estimators=5000,
        averaging="usual",
    )

    ds.submission["isFraud"] = result_dict_lgb["prediction"]
    return None