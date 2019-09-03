import gc
import logging
import os
import warnings
from datetime import datetime

import click
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, StratifiedKFold

from src.config import read_configuration, write_params
from src.dataset.make_dataset import Dataset
from src.features.build_features import (
    build_processed_dataset,
    convert_category_cols_lgb,
)
from src.model.split import TimeSeriesSplit, impute_mean
from src.model.train import (
    clf_lgb,
    clf_logistic,
    clf_xgb,
    clf_catboost,
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
    preprocessors = {"lgb": convert_category_cols_lgb}
    preprocessors_fold = {
        "logistic": impute_mean,
        "lgb": None,
        "xgb": None,
        "catboost": None,
    }
    modellers = {
        "logistic": clf_logistic,
        "lgb": clf_lgb,
        "xgb": clf_xgb,
        "catboost": clf_catboost,
    }

    # Read JSON conf for parameters
    conf = read_configuration(key)
    classifier = conf["classifier"]
    params = conf["params"]

    logger.info("Begin run experiment")

    logger.info("Loading dataset")
    ds = Dataset()
    ds.load_dataset(version)

    logger.info("Preprocessing data")
    build_processed_dataset(ds)
    if classifier in preprocessors:
        preprocessors[classifier](ds)
    gc.collect()

    logger.info(f"Building {classifier} model")
    folds = TimeSeriesSplit(n_splits=5)
    result = run_train_predict(
        ds, modellers[classifier], params, folds, preprocessors_fold[classifier]
    )

    if conf["save_predictions"]:
        path_to_preds = f"{key}_{time_experiment}"
        logger.info(f"Saving {key} predictions to {path_to_preds}")
        ds.submission["isFraud"] = result["prediction"]
        ds.write_submission(path_to_preds)

    if conf["save_models"]:
        path_to_models = get_root_dir() / f"models/{key}_{time_experiment}"
        logger.info(f"Saving raw models to {path_to_models}")

        os.mkdir(path_to_models)
        write_params(params, path_to_models / "params.json")
        for i, model in enumerate(result["models"]):
            save_model(model, path_to_models / f"fold_{i}")
        open(path_to_models / "_SUCCESS", "a").close()

    logger.info("End run experiment")


if __name__ == "__main__":
    run_experiment()
