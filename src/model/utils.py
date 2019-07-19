from path import Path
import pickle


def save_model(clf, name):
    folder = Path("../../models")
    pickle.dump(clf, open(folder / name, "wb"))


def load_model(name):
    folder = Path("../../models")
    return pickle.load(open(folder / name, "rb"))