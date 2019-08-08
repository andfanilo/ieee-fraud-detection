import gc
import click

from src.dataset.make_dataset import Dataset
from src.features.build_features import build_processed_dataset, drop_cols
from src.model.train_model import train_xgb, train_xgb_folds, train_lgb_folds

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--version", type=str, default="", help="Dataset version to load")
@click.option("--name", type=str, default="test", help="Submission name")
def run_experiment(version, name):
    logger.info("Begin run experiment")
    logger.info("Loading dataset")

    ds = Dataset()
    ds.load_dataset(version)

    logger.info("Preprocessing data")
    build_processed_dataset(ds)
    gc.collect()

    logger.info("Building LGB model without drop columns")
    result = train_lgb_folds(ds)
    ds.submission["isFraud"] = result["prediction"]
    ds.write_submission("lgb_folds")

    # logger.info("Building XGB model")
    # result = train_xgb_folds(ds)
    # ds.submission["isFraud"] = result["prediction"]
    # ds.write_submission("xgb_folds")

    # logger.info("Building XGB model")
    # result = train_xgb(ds)
    # ds.submission["isFraud"] = result["prediction"]
    # ds.write_submission("xgb")

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment(None, "main_submission")
