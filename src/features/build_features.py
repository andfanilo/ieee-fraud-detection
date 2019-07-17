from sklearn import preprocessing


def label_encode(ds):
    """
    Apply label encoder to ds.X_train and ds.X_test in dataset ds

    input: a Dataset
    output: (train, test)
    """
    for f in ds.X_train.columns:
        if ds.X_train[f].dtype == "object" or ds.X_test[f].dtype == "object":
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(ds.X_train[f].values) + list(ds.X_test[f].values))
            ds.X_train[f] = lbl.transform(list(ds.X_train[f].values))
            ds.X_test[f] = lbl.transform(list(ds.X_test[f].values))
