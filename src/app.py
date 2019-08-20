import gc
import logging
import warnings

import click
from sklearn.model_selection import train_test_split
from src.dataset.make_dataset import Dataset
from src.features.build_features import (
    build_processed_dataset,
    convert_category_cols_lgb,
)
from src.model.train_model import (
    train_lgb_folds,
    train_xgb,
    train_xgb_folds,
    train_cat_folds,
)

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename="./data/submissions/latest_run.log",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--version", type=str, default="", help="Dataset version to load")
@click.option(
    "--model",
    type=click.Choice(["simple", "xgb", "lgb", "cat"]),
    default="lgb",
    help="Type of model to run",
)
def run_experiment(version, model):
    logger.info("Begin run experiment")
    logger.info("Loading dataset")

    ds = Dataset()
    ds.load_dataset(version)

    logger.info("Preprocessing data")
    build_processed_dataset(ds)
    gc.collect()

    if model == "simple":
        logger.info("Building simple XGB model")
        X_train, X_valid, y_train, y_valid = train_test_split(
            ds.X_train, ds.y_train, test_size=0.3, random_state=0
        )
        result = train_xgb(
            X_train,
            X_valid,
            y_train,
            y_valid,
            ds.X_train.columns,
            {
                "eta": 0.05,
                "max_depth": 9,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "gamma": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "verbosity": 0,
                "random_state": 1337,
                "nthread": -1,
            },
        )
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission("xgb")

    elif model == "xgb":
        logger.info("Building XGB model folds")
        result = train_xgb_folds(ds)
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission("xgb_folds")

    elif model == "lgb":
        convert_category_cols_lgb(ds)

        logger.info("Building LGB model folds")
        result = train_lgb_folds(ds)
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission("lgb_folds")

    elif model == "cat":
        logger.info("Building Catboost model folds")
        result = train_cat_folds(ds)
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission("lgb_folds")

    else:
        return

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment()
