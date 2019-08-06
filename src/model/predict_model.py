from src.utils import get_root_dir


def make_predictions(ds, clf):
    ds.submission["isFraud"] = clf.predict_proba(ds.X_test)[:, 1]


def write_submission(ds, filename):
    sub_folder = get_root_dir() / "data/submissions"
    ds.submission.reset_index().to_csv(sub_folder / filename + ".csv", index=False)
