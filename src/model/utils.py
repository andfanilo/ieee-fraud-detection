import pickle

import numpy as np
from numba import jit
from src.utils import get_root_dir


def save_model(clf, name):
    folder = get_root_dir() / "models"
    pickle.dump(clf, open(folder / name + ".pickle.dat", "wb"))


def load_model(name):
    folder = get_root_dir() / "models"
    return pickle.load(open(folder / name + ".pickle.dat", "rb"))


@jit(nopython=True)
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc


@jit(nopython=True)
def fast_auc_weight(y_true, y_prob, w):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/80182
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        w_i = w[i]
        nfalse += (1 - y_i) * w_i
        auc += y_i * nfalse * w_i
    auc /= nfalse * (n - nfalse)
    return auc


def eval_auc_lgb(preds, dtrain):
    """
    Fast auc eval function for lgb.
    """
    labels = dtrain.get_label()
    return "fast auc", fast_auc(labels, preds), True


def eval_auc_xgb(preds, dtrain):
    """
    Fast auc eval function for xgb.
    """
    labels = dtrain.get_label()
    return "fast auc", fast_auc(labels, preds)


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
