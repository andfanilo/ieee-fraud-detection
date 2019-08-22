import gc
import logging
import warnings

import click
from src.dataset.make_dataset import Dataset
from src.features.build_features import (
    build_processed_dataset,
    convert_category_cols_lgb,
)
from src.model.train_model import (
    train_lgb_folds,
    train_xgb_folds,
    train_cat_folds,
    train_logistic_folds,
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
    type=click.Choice(["xgb", "lgb", "cat", "logistic"]),
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

    if model == "xgb":
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
        ds.write_submission("cat_folds")

    elif model == "logistic":
        ds.X_train = ds.X_train.fillna(-1)
        ds.X_test = ds.X_test.fillna(-1)

        logger.info("Building Logistic model folds")
        result = train_logistic_folds(ds)
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission("logistic_folds")

    else:
        return

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment()
