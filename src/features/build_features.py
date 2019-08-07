import numpy as np
from sklearn import preprocessing
import datetime

import logging

logger = logging.getLogger(__name__)


def convert_category_cols(ds):
    """
    Only necessary for LGB if doing automatic category
    """
    for col in ds.X_train.columns:
        # Work on strings to categorical if number of unique values is less than 50%
        if ds.X_train[col].dtype == object or ds.X_test[col].dtype == object:
            num_unique_values = len(
                set(list(ds.X_train[col].values) + list(ds.X_test[col].values))
            )
            num_total_values = len(
                list(ds.X_train[col].values) + list(ds.X_test[col].values)
            )
            if num_unique_values / num_total_values < 0.5:
                ds.X_train[col] = ds.X_train[col].astype("category")
                ds.X_test[col] = ds.X_test[col].astype("category")

        ## we also know all categorical columns already xD
        if col in ds.get_categorical_cols():
            ds.X_train[col] = ds.X_train[col].astype("category")
            ds.X_test[col] = ds.X_test[col].astype("category")


def create_date(ds, start_date="2017-12-01"):
    """
    Preprocess TransactionDT
    """
    startdate = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    ds.X_train["TransactionDT"] = ds.X_train["TransactionDT"].apply(
        lambda x: (startdate + datetime.timedelta(seconds=x))
    )
    ds.X_test["TransactionDT"] = ds.X_test["TransactionDT"].apply(
        lambda x: (startdate + datetime.timedelta(seconds=x))
    )


def label_encode(ds):
    """
    Apply label encoder to ds.X_train and ds.X_test categorical columns

    input: a Dataset
    output: (train, test)
    """
    # for col in ds.get_categorical_cols():
    for col in ds.X_train.columns:
        if ds.X_train[col].dtype == object or ds.X_test[col].dtype == object:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(ds.X_train[col].values) + list(ds.X_test[col].values))
            ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))
            ds.X_test[col] = lbl.transform(list(ds.X_test[col].values))


def aggregate_cols(ds):
    ds.X_train["TransactionAmt_to_mean_card1"] = ds.X_train[
        "TransactionAmt"
    ] / ds.X_train.groupby(["card1"])["TransactionAmt"].transform("mean")
    ds.X_train["TransactionAmt_to_mean_card4"] = ds.X_train[
        "TransactionAmt"
    ] / ds.X_train.groupby(["card4"])["TransactionAmt"].transform("mean")
    ds.X_train["TransactionAmt_to_std_card1"] = ds.X_train[
        "TransactionAmt"
    ] / ds.X_train.groupby(["card1"])["TransactionAmt"].transform("std")
    ds.X_train["TransactionAmt_to_std_card4"] = ds.X_train[
        "TransactionAmt"
    ] / ds.X_train.groupby(["card4"])["TransactionAmt"].transform("std")

    ds.X_test["TransactionAmt_to_mean_card1"] = ds.X_test[
        "TransactionAmt"
    ] / ds.X_test.groupby(["card1"])["TransactionAmt"].transform("mean")
    ds.X_test["TransactionAmt_to_mean_card4"] = ds.X_test[
        "TransactionAmt"
    ] / ds.X_test.groupby(["card4"])["TransactionAmt"].transform("mean")
    ds.X_test["TransactionAmt_to_std_card1"] = ds.X_test[
        "TransactionAmt"
    ] / ds.X_test.groupby(["card1"])["TransactionAmt"].transform("std")
    ds.X_test["TransactionAmt_to_std_card4"] = ds.X_test[
        "TransactionAmt"
    ] / ds.X_test.groupby(["card4"])["TransactionAmt"].transform("std")

    ds.X_train["id_02_to_mean_card1"] = ds.X_train["id_02"] / ds.X_train.groupby(
        ["card1"]
    )["id_02"].transform("mean")
    ds.X_train["id_02_to_mean_card4"] = ds.X_train["id_02"] / ds.X_train.groupby(
        ["card4"]
    )["id_02"].transform("mean")
    ds.X_train["id_02_to_std_card1"] = ds.X_train["id_02"] / ds.X_train.groupby(
        ["card1"]
    )["id_02"].transform("std")
    ds.X_train["id_02_to_std_card4"] = ds.X_train["id_02"] / ds.X_train.groupby(
        ["card4"]
    )["id_02"].transform("std")

    ds.X_test["id_02_to_mean_card1"] = ds.X_test["id_02"] / ds.X_test.groupby(
        ["card1"]
    )["id_02"].transform("mean")
    ds.X_test["id_02_to_mean_card4"] = ds.X_test["id_02"] / ds.X_test.groupby(
        ["card4"]
    )["id_02"].transform("mean")
    ds.X_test["id_02_to_std_card1"] = ds.X_test["id_02"] / ds.X_test.groupby(["card1"])[
        "id_02"
    ].transform("std")
    ds.X_test["id_02_to_std_card4"] = ds.X_test["id_02"] / ds.X_test.groupby(["card4"])[
        "id_02"
    ].transform("std")

    ds.X_train["D15_to_mean_card1"] = ds.X_train["D15"] / ds.X_train.groupby(["card1"])[
        "D15"
    ].transform("mean")
    ds.X_train["D15_to_mean_card4"] = ds.X_train["D15"] / ds.X_train.groupby(["card4"])[
        "D15"
    ].transform("mean")
    ds.X_train["D15_to_std_card1"] = ds.X_train["D15"] / ds.X_train.groupby(["card1"])[
        "D15"
    ].transform("std")
    ds.X_train["D15_to_std_card4"] = ds.X_train["D15"] / ds.X_train.groupby(["card4"])[
        "D15"
    ].transform("std")

    ds.X_test["D15_to_mean_card1"] = ds.X_test["D15"] / ds.X_test.groupby(["card1"])[
        "D15"
    ].transform("mean")
    ds.X_test["D15_to_mean_card4"] = ds.X_test["D15"] / ds.X_test.groupby(["card4"])[
        "D15"
    ].transform("mean")
    ds.X_test["D15_to_std_card1"] = ds.X_test["D15"] / ds.X_test.groupby(["card1"])[
        "D15"
    ].transform("std")
    ds.X_test["D15_to_std_card4"] = ds.X_test["D15"] / ds.X_test.groupby(["card4"])[
        "D15"
    ].transform("std")

    ds.X_train["D15_to_mean_addr1"] = ds.X_train["D15"] / ds.X_train.groupby(["addr1"])[
        "D15"
    ].transform("mean")
    ds.X_train["D15_to_mean_card4"] = ds.X_train["D15"] / ds.X_train.groupby(["card4"])[
        "D15"
    ].transform("mean")
    ds.X_train["D15_to_std_addr1"] = ds.X_train["D15"] / ds.X_train.groupby(["addr1"])[
        "D15"
    ].transform("std")
    ds.X_train["D15_to_std_card4"] = ds.X_train["D15"] / ds.X_train.groupby(["card4"])[
        "D15"
    ].transform("std")

    ds.X_test["D15_to_mean_addr1"] = ds.X_test["D15"] / ds.X_test.groupby(["addr1"])[
        "D15"
    ].transform("mean")
    ds.X_test["D15_to_mean_card4"] = ds.X_test["D15"] / ds.X_test.groupby(["card4"])[
        "D15"
    ].transform("mean")
    ds.X_test["D15_to_std_addr1"] = ds.X_test["D15"] / ds.X_test.groupby(["addr1"])[
        "D15"
    ].transform("std")
    ds.X_test["D15_to_std_card4"] = ds.X_test["D15"] / ds.X_test.groupby(["card4"])[
        "D15"
    ].transform("std")


def parse_emails(ds):
    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_df-579654
    emails = {
        "gmail": "google",
        "att.net": "att",
        "twc.com": "spectrum",
        "scranton.edu": "other",
        "optonline.net": "other",
        "hotmail.co.uk": "microsoft",
        "comcast.net": "other",
        "yahoo.com.mx": "yahoo",
        "yahoo.fr": "yahoo",
        "yahoo.es": "yahoo",
        "charter.net": "spectrum",
        "live.com": "microsoft",
        "aim.com": "aol",
        "hotmail.de": "microsoft",
        "centurylink.net": "centurylink",
        "gmail.com": "google",
        "me.com": "apple",
        "earthlink.net": "other",
        "gmx.de": "other",
        "web.de": "other",
        "cfl.rr.com": "other",
        "hotmail.com": "microsoft",
        "protonmail.com": "other",
        "hotmail.fr": "microsoft",
        "windstream.net": "other",
        "outlook.es": "microsoft",
        "yahoo.co.jp": "yahoo",
        "yahoo.de": "yahoo",
        "servicios-ta.com": "other",
        "netzero.net": "other",
        "suddenlink.net": "other",
        "roadrunner.com": "other",
        "sc.rr.com": "other",
        "live.fr": "microsoft",
        "verizon.net": "yahoo",
        "msn.com": "microsoft",
        "q.com": "centurylink",
        "prodigy.net.mx": "att",
        "frontier.com": "yahoo",
        "anonymous.com": "other",
        "rocketmail.com": "yahoo",
        "sbcglobal.net": "att",
        "frontiernet.net": "yahoo",
        "ymail.com": "yahoo",
        "outlook.com": "microsoft",
        "mail.com": "other",
        "bellsouth.net": "other",
        "embarqmail.com": "centurylink",
        "cableone.net": "other",
        "hotmail.es": "microsoft",
        "mac.com": "apple",
        "yahoo.co.uk": "yahoo",
        "netzero.com": "other",
        "yahoo.com": "yahoo",
        "live.com.mx": "microsoft",
        "ptd.net": "other",
        "cox.net": "other",
        "aol.com": "aol",
        "juno.com": "other",
        "icloud.com": "apple",
    }
    us_emails = ["gmail", "net", "edu"]
    for c in ["P_emaildomain", "R_emaildomain"]:
        ds.X_train[c + "_bin"] = ds.X_train[c].map(emails)
        ds.X_test[c + "_bin"] = ds.X_test[c].map(emails)

        ds.X_train[c + "_suffix"] = ds.X_train[c].map(lambda x: str(x).split(".")[-1])
        ds.X_test[c + "_suffix"] = ds.X_test[c].map(lambda x: str(x).split(".")[-1])

        ds.X_train[c + "_suffix"] = ds.X_train[c + "_suffix"].map(
            lambda x: x if str(x) not in us_emails else "us"
        )
        ds.X_test[c + "_suffix"] = ds.X_test[c + "_suffix"].map(
            lambda x: x if str(x) not in us_emails else "us"
        )


def clean_inf_nan(ds):
    # by https://www.kaggle.com/dimartinot
    ds.X_train.replace([np.inf, -np.inf], np.nan)
    ds.X_test.replace([np.inf, -np.inf], np.nan)


def fill_nan(ds):
    ds.X_train.fillna(-999, inplace=True)
    ds.X_test.fillna(-999, inplace=True)


def drop_cols(ds):
    one_value_cols = [
        col for col in ds.X_train.columns if ds.X_train[col].nunique() <= 1
    ]
    one_value_cols_test = [
        col for col in ds.X_test.columns if ds.X_test[col].nunique() <= 1
    ]

    many_null_cols = [
        col
        for col in ds.X_train.columns
        if ds.X_train[col].isnull().sum() / ds.X_train.shape[0] > 0.9
    ]
    many_null_cols_test = [
        col
        for col in ds.X_test.columns
        if ds.X_test[col].isnull().sum() / ds.X_test.shape[0] > 0.9
    ]
    big_top_value_cols = [
        col
        for col in ds.X_train.columns
        if ds.X_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]
    big_top_value_cols_test = [
        col
        for col in ds.X_test.columns
        if ds.X_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]

    cols_to_drop = list(
        set(
            many_null_cols
            + many_null_cols_test
            + big_top_value_cols
            + big_top_value_cols_test
            + one_value_cols
            + one_value_cols_test
        )
    )
    logger.info("Will drop the following columns")
    logger.info(",".join(cols_to_drop))

    ds.X_train = ds.X_train.drop(cols_to_drop, axis=1)
    ds.X_test = ds.X_test.drop(cols_to_drop, axis=1)


def build_processed_dataset(ds):
    clean_inf_nan(ds)
    fill_nan(ds)
    parse_emails(ds)
    label_encode(ds)
    # convert_category_cols(ds)
    aggregate_cols(ds)

    # drop_cols(ds)
