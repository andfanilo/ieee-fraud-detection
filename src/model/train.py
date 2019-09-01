import os
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from src.model.utils import eval_auc, group_mean_log_mae
from src.model.split import TimeSeriesSplit
from src.utils import print_colored_green, get_root_dir

logger = logging.getLogger(__name__)


def clf_logistic(X_train, y_train, X_valid, y_valid, X_test, params):
    model = LogisticRegression(random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    score = metrics.roc_auc_score(y_valid, y_pred_valid)
    print(f"Fold result : AUC {score:.4f}.")

    y_pred = model.predict_proba(X_test)[:, 1]
    return model, y_pred_valid, y_pred


def clf_lgb(X_train, y_train, X_valid, y_valid, X_test, params):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric=eval_auc,
        verbose=100,
    )

    y_pred_valid = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[
        :, 1
    ]
    y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
    return model, y_pred_valid, y_pred


def run_train_predict(ds, clf, params, folds, preprocess_fold=None, averaging="usual"):
    X = ds.X_train
    X_test = ds.X_test
    y = ds.y_train
    n_splits = folds.n_splits

    # Prepare output, list of scores on folds
    result_dict = {}
    result_dict["models"] = []
    scores = []
    feature_importance = pd.DataFrame()

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

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f"Fold {fold_n + 1} started at {time.ctime()}")
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = (X.iloc[train_index], X.iloc[valid_index])
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # fold preprocessing
        if preprocess_fold:
            X_train, y_train, X_valid, y_valid, X_test = preprocess_fold(
                X_train, y_train, X_valid, y_valid, X_test
            )

        # train & predict validation / test set
        model, y_pred_valid, y_pred = clf(
            X_train, y_train, X_valid, y_valid, X_test, params
        )

        # Save models to return
        result_dict["models"].append(model)

        # Save out-of-fold predictions
        if averaging == "usual":
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics.roc_auc_score(y_valid, y_pred_valid))
            prediction += y_pred.reshape(-1, 1)

        elif averaging == "rank":
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics.roc_auc_score(y_valid, y_pred_valid))
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

    prediction /= n_splits
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    logger.info(f"CV mean score: {mean_score:.4f}, std: {std_score:.4f}.")
    print_colored_green(f"CV mean score: {mean_score:.4f}, std: {std_score:.4f}.")

    result_dict["oof"] = oof
    result_dict["prediction"] = prediction
    result_dict["scores"] = scores

    return result_dict
