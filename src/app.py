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
    train_cat_folds,
    train_lgb_folds,
    train_logistic_folds,
    train_xgb_folds,
)
from src.utils import seed_everything

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
    seed_everything(0)

    preprocessor = {"lgb": convert_category_cols_lgb}
    modeller = {
        "xgb": train_xgb_folds,
        "lgb": train_lgb_folds,
        "cat": train_cat_folds,
        "logistic": train_logistic_folds,
    }

    logger.info("Begin run experiment")

    logger.info("Loading dataset")
    ds = Dataset()
    ds.load_dataset(version)

    logger.info("Preprocessing data")
    build_processed_dataset(ds)
    if model in preprocessor:
        preprocessor[model](ds)
    gc.collect()

    logger.info(f"Building {model} model")
    result = modeller[model](ds)

    logger.info(f"Saving {model} predictions")
    ds.submission["isFraud"] = result["prediction"]
    ds.write_submission(f"{model}_preds")

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment()
