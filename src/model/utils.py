from src.utils import get_root_dir
import pickle


def save_model(clf, name):
    folder = get_root_dir() / "models"
    pickle.dump(clf, open(folder / name, "wb"))


def load_model(name):
    folder = get_root_dir() / "models"
    return pickle.load(open(folder / name, "rb"))