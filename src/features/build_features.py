from sklearn import preprocessing
import datetime


def convert_categories(ds):
    for col in ds.X_train.columns:
    # Work on strings to categorical if number of unique values is less than 50%
        if ds.test_loaded:
            if ds.X_train[col].dtype == object or ds.X_test[col].dtype == object:
                num_unique_values = len((list(ds.X_train[col].values) + list(ds.X_test[col].values)).unique())
                num_total_values = len(list(ds.X_train[col].values) + list(ds.X_test[col].values))
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')

            ## we also know all categorical columns already xD
            if col in ds.get_categorical_cols():
                ds.X_train[col] = ds.X_train[col].astype('category')
                ds.X_test[col] = ds.X_test[col].astype('category')

        else:
            if ds.X_train[col].dtype == object:
                num_unique_values = len(ds.X_train[col].unique())
                num_total_values = len(ds.X_train[col])
                if num_unique_values / num_total_values < 0.5:
                    ds.X_train[col] = ds.X_train[col].astype('category')

            if col in ds.get_categorical_cols():
                ds.X_train[col] = ds.X_train[col].astype('category')



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
    for col in ds.X_train.columns:
        ds.X_train.fillna(-999, inplace=True)
        if ds.test_loaded:
            ds.X_test.fillna(-999, inplace=True)
            if ds.X_train[col].dtype == "object" or ds.X_test[col].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(ds.X_train[col].values) + list(ds.X_test[col].values))
                ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))
                ds.X_test[col] = lbl.transform(list(ds.X_test[col].values))
        else:
            if ds.X_train[col].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(ds.X_train[col].values))
                ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))


def build_features(ds):
    create_date(ds)