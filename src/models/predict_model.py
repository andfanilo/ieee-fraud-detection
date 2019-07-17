from path import Path

def write_submission(ds, clf, filename):
    sub_folder = Path("../../data/submissions")
    ds.sample_submission['isFraud'] = clf.predict_proba(ds.X_test)[:,1]
    ds.sample_submission.to_csv(sub_folder / filename)