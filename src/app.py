import gc
import click

from src.dataset.make_dataset import Dataset
from src.features.build_features import build_processed_dataset
from src.model.train_model import train_xgb_folds, train_lgb_folds
from src.model.predict_model import write_submission
from src.model.utils import save_model

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--version", type=str, default=None, help="Dataset version to load")
@click.option("--name", type=str, default="test", help="Submission name")
def run_experiment(version, name):
    logger.info("Begin run experiment")
    logger.info("Loading dataset")
    ds = Dataset()
    ds.load_dataset(version)
    # ds.load_raw(int(version))
    # ds.save_dataset(version)

    logger.info("Preprocessing data")
    build_processed_dataset(ds)
    gc.collect()

    logger.info("Building ML model")
    train_xgb_folds(ds)
    # train_lgb_folds(ds)
    write_submission(ds, name)

    # clf = train_xgb(ds)
    # make_predictions(ds, clf)
    # save_model(clf, name)

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment(None, "main_submission")
