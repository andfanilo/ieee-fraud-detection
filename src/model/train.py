import logging
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from src.dataset.data import Fold
from src.features.process_fold import process_fold
from src.model.split import TimeSeriesSplit
from src.model.utils import eval_auc_lgb, eval_auc_xgb
from src.utils import get_root_dir, print_colored_green

logger = logging.getLogger(__name__)


def clf_logistic(X_train, y_train, X_valid, y_valid, X_test, params):
    model = LogisticRegression(random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    score = metrics.roc_auc_score(y_valid, y_pred_valid)
    print(f"Fold result : AUC {score:.4f}.")

    y_pred = model.predict_proba(X_test)[:, 1]
    return model, y_pred_valid, y_pred


def clf_xgb(X_train, y_train, X_valid, y_valid, X_test, params):
    n_estimators = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]
    params.pop("categorical_feature")

    columns = X_train.columns
    train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=columns)
    valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=columns)

    watchlist = [(train_data, "train"), (valid_data, "valid_data")]
    model = xgb.train(
        params=params,
        dtrain=train_data,
        evals=watchlist,
        # feval=eval_auc_xgb, # put disable_default_eval_metric:"1" in params
        verbose_eval=100,
        num_boost_round=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    y_pred_valid = model.predict(
        xgb.DMatrix(X_valid, feature_names=columns), ntree_limit=model.best_ntree_limit
    )
    y_pred = model.predict(
        xgb.DMatrix(X_test, feature_names=columns), ntree_limit=model.best_ntree_limit
    )

    return model, y_pred_valid, y_pred


def clf_lgb(X_train, y_train, X_valid, y_valid, X_test, params):
    n_estimators = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]

    categorical_feature = params["categorical_feature"]
    logger.info("Following columns considered categorical for training")
    logger.info(", ".join(categorical_feature))

    train_data = lgb.Dataset(data=X_train, label=y_train)
    valid_data = lgb.Dataset(data=X_valid, label=y_valid)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        # feval=eval_auc_lgb, # put metric:"None" in params
        categorical_feature=categorical_feature,
        verbose_eval=100,
        num_boost_round=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return model, y_pred_valid, y_pred


def clf_catboost(X_train, y_train, X_valid, y_valid, X_test, params):
    # TODO : won't work for kfold but catboost kfold is ultra long..
    categorical_feature = params.pop("categorical_feature")
    logger.info("Following columns considered categorical for training")
    logger.info(", ".join(categorical_feature))

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=categorical_feature,
        verbose=100,
    )

    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    y_pred = model.predict_proba(X_test)[:, 1]

    return model, y_pred_valid, y_pred


def run_train_predict(ds, clf, params, folds, preprocess_fold=None, averaging="usual"):
    """Run clf function on ds with given folds
    """
    X = ds.X_train
    X_test = ds.X_test
    y = ds.y_train

    category_cols = ds.categorical_cols
    n_splits = folds.get_n_splits()

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

        fold = Fold(X_train, y_train, X_valid, y_valid, X_test, category_cols)

        # fold preprocessing
        process_fold(fold)

        # automatically extract categorical features for catboost, or chosoe for lgb
        if "categorical_feature" not in params:
            params["categorical_feature"] = fold.categorical_cols

        # train & predict validation / test set
        model, y_pred_valid, y_pred = clf(
            fold.X_train, fold.y_train, fold.X_valid, fold.y_valid, fold.X_test, params
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
