from src.utils import get_root_dir

def write_submission(ds, clf, filename):
    sub_folder = get_root_dir() / "data/submissions"
    ds.sample_submission['isFraud'] = clf.predict_proba(ds.X_test)[:,1]
    ds.sample_submission.reset_index().to_csv(sub_folder / filename, index=False)