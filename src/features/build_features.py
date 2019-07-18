from sklearn import preprocessing
import datetime


def create_date(ds, start_date='2017-12-01'):
    """
    Preprocess TransactionDT
    """
    startdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    ds.X_train['TransactionDT'] = ds.X_train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    if ds.test_loaded:
        ds.X_test['TransactionDT'] = ds.X_test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


def label_encode(ds):
    """
    Apply label encoder to ds.X_train and ds.X_test in dataset ds

    input: a Dataset
    output: (train, test)
    """
    
    for f in ds.X_train.columns:
        ds.X_train.fillna(-999, inplace=True)
        if ds.test_loaded:
            ds.X_test.fillna(-999, inplace=True)
            if ds.X_train[f].dtype == "object" or ds.X_test[f].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(ds.X_train[f].values) + list(ds.X_test[f].values))
                ds.X_train[f] = lbl.transform(list(ds.X_train[f].values))
                ds.X_test[f] = lbl.transform(list(ds.X_test[f].values))
        else:
            if ds.X_train[f].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(ds.X_train[f].values))
                ds.X_train[f] = lbl.transform(list(ds.X_train[f].values))


def build_features(ds):
    create_date(ds)