import gc
import click

from src.dataset.make_dataset import Dataset
from src.features.build_features import build_processed_dataset
from src.model.train_model import train_lgb_folds
from src.model.predict_model import write_submission
from src.model.utils import save_model

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--nrows_train", type=int, default=None, help="Number of training rows")
@click.option("--name", type=str, default="submission", help="Submission name")
def run_experiment(nrows_train, name):
    logger.info("Begin run experiment")
    logger.info("Loading dataset")
    ds = Dataset()
    ds.load_dataset("processed")

    # logger.info("Preprocessing data")
    # build_processed_dataset(ds)
    # ds.save_dataset("processed")
    gc.collect()

    logger.info("Building ML model")
    clf = train_lgb_folds(ds)
    write_submission(ds, clf, name)
    save_model(clf, name)

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment(1000, "main_submission")
