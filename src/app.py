import gc
import logging
import os
import warnings
from datetime import datetime

import click
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, StratifiedKFold
from src.config import read_configuration, write_params
from src.dataset.data import Dataset
from src.features.build_features import build_processed_dataset
from src.model.split import CustomDateSplitter
from src.model.train import (
    clf_catboost,
    clf_lgb,
    clf_logistic,
    clf_xgb,
    run_train_predict,
)
from src.model.utils import save_model
from src.utils import get_root_dir, seed_everything

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    # filename="./data/submissions/latest_run.log",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--version", type=str, default="", help="Dataset version to load")
@click.option("--key", type=str, help="Experiment to run, see keys in conf.json file")
def run_experiment(version, key):
    time_experiment = datetime.now().strftime("%m%d%Y_%H%M")
    seed_everything(0)

    # Predefine functions
    modellers = {
        "logistic": clf_logistic,
        "lgb": clf_lgb,
        "xgb": clf_xgb,
        "catboost": clf_catboost,
    }

    ########################### READ PARAMETERS
    conf = read_configuration(key)
    classifier = conf["classifier"]
    params = conf["params"]
    split = conf["splits"]

    logger.info("Begin run experiment")

    ########################### LOADING DATASET
    logger.info("Loading dataset")
    ds = Dataset()
    ds.load_dataset(version)

    ########################### BUILD CROSS VALIDATION STRATEGY
    logger.info("Build folds")
    date_ranges = [
        # [["2018-01-01", "2018-05-31"], ["2017-12-01", "2017-12-31"]],
        [["2017-12-01", "2018-04-15"], ["2018-05-01", "2018-05-31"]]
    ]
    splits = {
        "holdout": CustomDateSplitter(ds.X_train["TransactionDT"], date_ranges),
        "kfold": KFold(n_splits=6, shuffle=False),
    }
    folds = splits[split]

    ########################### PREPROCESSING DATA
    logger.info("Preprocessing data")
    build_processed_dataset(ds)

    gc.collect()

    ########################### TRAIN MODEL
    logger.info(f"Building {classifier} model")
    result = run_train_predict(ds, modellers[classifier], params, folds)

    ########################### SAVING
    if conf["save_predictions"]:
        path_to_preds = f"{key}_{time_experiment}"
        logger.info(f"Saving {key} predictions to {path_to_preds}")
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission(path_to_preds)

    if conf["save_models"]:
        ds.save_dataset(f"{key}_processed")
        path_to_models = get_root_dir() / f"models/{key}_{time_experiment}"
        logger.info(f"Saving raw models to {path_to_models}")

        os.mkdir(path_to_models)
        write_params(params, path_to_models / "params.json")
        for i, model in enumerate(result["models"]):
            save_model(model, path_to_models / f"fold_{i}")
        open(path_to_models / "_SUCCESS", "a").close()

    logger.info("End run experiment")


@click.command()
def run_baseline():
    time_experiment = datetime.now().strftime("%m%d%Y_%H%M")
    seed_everything(0)

    ds = Dataset()
    ds.load_dataset()

    from sklearn.preprocessing import LabelEncoder

    nan_constant = -999
    for col, col_type in ds.X_train.dtypes.iteritems():
        if col_type == "object":
            ds.X_train[col] = ds.X_train[col].fillna(nan_constant)
            ds.X_test[col] = ds.X_test[col].fillna(nan_constant)

            lbl = LabelEncoder()
            lbl.fit(list(ds.X_train[col].values) + list(ds.X_test[col].values))
            ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))
            ds.X_test[col] = lbl.transform(list(ds.X_test[col].values))

            if nan_constant in lbl.classes_:
                nan_transformed = lbl.transform([nan_constant])[0]
                ds.X_train.loc[ds.X_train[col] == nan_transformed, col] = np.nan
                ds.X_test.loc[ds.X_test[col] == nan_transformed, col] = np.nan

        if col in ds.categorical_cols:
            ds.X_train[col] = ds.X_train[col].fillna(-1).astype("category")
            ds.X_test[col] = ds.X_test[col].fillna(-1).astype("category")

    lgb_params = {
        "n_estimators": 50000,
        "early_stopping_rounds": 200,
        "num_leaves": 256,
        "learning_rate": 0.03,
        "max_depth": 9,
        "objective": "binary",
        "metric": "auc",
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": 5.5,  #  mimic a fraud rate ~20% - 0.2*N_legit/N_fraud
        "boosting_type": "gbdt",
        "seed": 1337,
        "n_jobs": -1,
        "verbosity": -1,
    }

    folds = KFold(n_splits=5, random_state=0, shuffle=False)
    result = run_train_predict(ds, clf_lgb, lgb_params, folds, None)

    path_to_preds = f"baseline_lgb_{time_experiment}"
    ds.submission["isFraud"] = result["prediction"]
    ds.write_submission(path_to_preds)


if __name__ == "__main__":
    run_experiment("", "lgb")
