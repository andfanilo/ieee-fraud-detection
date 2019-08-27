import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.features.reduce_dimensions import VestaReducer
from src.features.utils import calc_smooth_mean

logger = logging.getLogger(__name__)


def clean_inf_nan(ds):
    # by https://www.kaggle.com/dimartinot
    # inf_cols_train = ds.X_train.columns.to_series()[np.isinf(ds.X_train).any()]
    # inf_cols_test = ds.X_test.columns.to_series()[np.isinf(ds.X_test).any()]
    ds.X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # logger.info("Following columns contained infinity values that were replaced by NAN")
    # logger.info(", ".join(list(set(inf_cols_train + inf_cols_test))))


def clean_noise_cards(ds, valid_counts=2):
    # https://www.kaggle.com/kyakovlev/ieee-ground-baseline
    # Reset values for "noise" card1 (account sampling)
    valid_card = ds.X_train["card1"].value_counts()
    valid_card = valid_card[valid_card > valid_counts]
    valid_card = list(valid_card.index)

    ds.X_train["card1"] = np.where(
        ds.X_train["card1"].isin(valid_card), ds.X_train["card1"], np.nan
    )
    ds.X_test["card1"] = np.where(
        ds.X_test["card1"].isin(valid_card), ds.X_test["card1"], np.nan
    )

    logger.info("Following card1 values were dropped")
    logger.info(", ".join([str(card) for card in valid_card]))


def transform_amount(ds):
    # Let's add some kind of client uID based on cardID ad addr columns
    # The value will be very specific for each client so we need to remove it
    # from final feature. But we can use it for aggregations.
    for df in [ds.X_train, ds.X_test]:
        df["uid"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
        df["uid2"] = (
            df["uid"].astype(str)
            + "_"
            + df["card3"].astype(str)
            + "_"
            + df["card5"].astype(str)
        )
        df["uid3"] = (
            df["uid2"].astype(str)
            + "_"
            + df["addr1"].astype(str)
            + "_"
            + df["addr2"].astype(str)
        )
        df["TransactionAmt"] = np.log1p(df["TransactionAmt"])

    # Check if the Transaction Amount is common or not (we can use freq encoding here)
    # In our dialog with a model we are telling to trust or not to these values
    ds.X_train["TransactionAmt_check"] = np.where(
        ds.X_train["TransactionAmt"].isin(ds.X_test["TransactionAmt"]), 1, 0
    )
    ds.X_test["TransactionAmt_check"] = np.where(
        ds.X_test["TransactionAmt"].isin(ds.X_train["TransactionAmt"]), 1, 0
    )


def parse_emails(ds):
    p = "P_emaildomain"
    r = "R_emaildomain"
    unknown = "email_not_provided"

    for df in [ds.X_train, ds.X_test]:
        df[p] = df[p].fillna(unknown)
        df[r] = df[r].fillna(unknown)
        df["email_check"] = np.where((df[p] == df[r]) & (df[p] != unknown), 1, 0)

        df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown_device").str.lower()
        df["DeviceInfo_device"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )
        df["DeviceInfo_version"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isnumeric()])
        )

        ########################### Device info 2
        df["id_30"] = df["id_30"].fillna("unknown_device").str.lower()
        df["id_30_device"] = df["id_30"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )
        df["id_30_version"] = df["id_30"].apply(
            lambda x: "".join([i for i in x if i.isnumeric()])
        )

        ########################### Browser
        df["id_31"] = df["id_31"].fillna("unknown_device").str.lower()
        df["id_31_device"] = df["id_31"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )

    logger.info("Emails were parsed")


def map_emails(ds):
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

    logger.info("Emails were mapped")


def parse_device(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    for df in [ds.X_train, ds.X_test]:
        df["device_name"] = df["DeviceInfo"].str.split("/", expand=True)[0]
        df["device_version"] = df["DeviceInfo"].str.split("/", expand=True)[1]

        df["OS_id_30"] = df["id_30"].str.split(" ", expand=True)[0]
        df["version_id_30"] = df["id_30"].str.split(" ", expand=True)[1]

        df["browser_id_31"] = df["id_31"].str.split(" ", expand=True)[0]
        df["version_id_31"] = df["id_31"].str.split(" ", expand=True)[1]

        df["screen_width"] = df["id_33"].str.split("x", expand=True)[0]
        df["screen_height"] = df["id_33"].str.split("x", expand=True)[1]

        df["id_34"] = df["id_34"].str.split(":", expand=True)[1]
        df["id_23"] = df["id_23"].str.split(":", expand=True)[1]

        df.loc[
            df["device_name"].str.contains("SM", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("SAMSUNG", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("GT-", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("Moto G", na=False), "device_name"
        ] = "Motorola"
        df.loc[
            df["device_name"].str.contains("Moto", na=False), "device_name"
        ] = "Motorola"
        df.loc[
            df["device_name"].str.contains("moto", na=False), "device_name"
        ] = "Motorola"
        df.loc[df["device_name"].str.contains("LG-", na=False), "device_name"] = "LG"
        df.loc[df["device_name"].str.contains("rv:", na=False), "device_name"] = "RV"
        df.loc[
            df["device_name"].str.contains("HUAWEI", na=False), "device_name"
        ] = "Huawei"
        df.loc[
            df["device_name"].str.contains("ALE-", na=False), "device_name"
        ] = "Huawei"
        df.loc[df["device_name"].str.contains("-L", na=False), "device_name"] = "Huawei"
        df.loc[df["device_name"].str.contains("Blade", na=False), "device_name"] = "ZTE"
        df.loc[df["device_name"].str.contains("BLADE", na=False), "device_name"] = "ZTE"
        df.loc[
            df["device_name"].str.contains("Linux", na=False), "device_name"
        ] = "Linux"
        df.loc[df["device_name"].str.contains("XT", na=False), "device_name"] = "Sony"
        df.loc[df["device_name"].str.contains("HTC", na=False), "device_name"] = "HTC"
        df.loc[df["device_name"].str.contains("ASUS", na=False), "device_name"] = "Asus"

        df.loc[
            df.device_name.isin(
                df.device_name.value_counts()[df.device_name.value_counts() < 200].index
            ),
            "device_name",
        ] = "Others"
        df["had_id"] = 1

    logger.info("IDs were splitted")


def build_transaction_features(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    # log of transaction amount.
    ds.X_train["TransactionAmt_Log"] = np.log(ds.X_train["TransactionAmt"])
    ds.X_test["TransactionAmt_Log"] = np.log(ds.X_test["TransactionAmt"])

    # decimal part of the transaction amount.
    ds.X_train["TransactionAmt_decimal"] = (
        (ds.X_train["TransactionAmt"] - ds.X_train["TransactionAmt"].astype(int)) * 1000
    ).astype(int)
    ds.X_test["TransactionAmt_decimal"] = (
        (ds.X_test["TransactionAmt"] - ds.X_test["TransactionAmt"].astype(int)) * 1000
    ).astype(int)

    logger.info("Built transaction features - log, decimal")


def build_date_features(ds, start_date="2017-11-30"):
    """
    Preprocess TransactionDT
    """
    # Temporary
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    for df in [ds.X_train, ds.X_test]:
        df["DT"] = df["TransactionDT"].apply(
            lambda x: (start_date + datetime.timedelta(seconds=x))
        )
        df["DT_M"] = (df["DT"].dt.year - 2017) * 12 + df["DT"].dt.month
        df["DT_W"] = (df["DT"].dt.year - 2017) * 52 + df["DT"].dt.weekofyear
        df["DT_D"] = (df["DT"].dt.year - 2017) * 365 + df["DT"].dt.dayofyear

        df["DT_hour"] = df["DT"].dt.hour
        df["DT_day_week"] = df["DT"].dt.dayofweek
        df["DT_day"] = df["DT"].dt.day

        # D9 is an hour
        df["D9"] = np.where(df["D9"].isna(), 0, 1)

    logger.info("Built date features day_of_week and hour")


def build_interaction_features(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    # Some arbitrary features interaction + those extracted from our own analysis
    random_intersection = [
        "id_02__id_20",
        "id_02__D8",
        "D11__DeviceInfo",
        "DeviceInfo__P_emaildomain",
        "P_emaildomain__C2",
        "card2__dist1",
        "card1__card5",
        "card2__id_20",
        "card5__P_emaildomain",
        "addr1__card1",
    ]
    my_intersections = ["C11__C13"]
    for feature in random_intersection + my_intersections:
        f1, f2 = feature.split("__")
        ds.X_train[feature] = (
            ds.X_train[f1].astype(str) + "_" + ds.X_train[f2].astype(str)
        )
        ds.X_test[feature] = ds.X_test[f1].astype(str) + "_" + ds.X_test[f2].astype(str)

    logger.info("Following columns were created for interaction")
    logger.info(", ".join(random_intersection + my_intersections))


def aggregate_cols(ds):
    # For our model current TransactionAmt is a noise
    # https://www.kaggle.com/kyakovlev/ieee-check-noise
    # (even if features importances are telling contrariwise)
    # There are many unique values and model doesn't generalize well
    # Lets do some aggregations
    i_cols = ["card1", "card2", "card3", "card5", "uid", "uid2", "uid3"]

    for col in i_cols:
        for agg_type in ["mean", "std"]:
            new_col_name = col + "_TransactionAmt_" + agg_type
            temp_df = pd.concat(
                [
                    ds.X_train[[col, "TransactionAmt"]],
                    ds.X_test[[col, "TransactionAmt"]],
                ]
            )
            temp_df = (
                temp_df.groupby([col])["TransactionAmt"]
                .agg([agg_type])
                .reset_index()
                .rename(columns={agg_type: new_col_name})
            )

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()

            ds.X_train[new_col_name] = ds.X_train[col].map(temp_df)
            ds.X_test[new_col_name] = ds.X_test[col].map(temp_df)

    logger.info("TransactionAmt was aggregated")


def binary_encoding(ds):
    """Encode all M columns except M4
    """
    i_cols = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]

    for df in [ds.X_train, ds.X_test]:
        df["M_sum"] = df[i_cols].sum(axis=1).astype(np.int8)
        df["M_na"] = df[i_cols].isna().sum(axis=1).astype(np.int8)

    logger.info("Following columns were binary encoded")
    logger.info(", ".join(i_cols))


def one_hot_encoding(ds):
    # https://markhneedham.com/blog/2017/07/05/pandasscikit-learn-get_dummies-testtrain-sets-valueerror-shapes-not-aligned/
    i_cols = []
    unknown = "?"

    for col in i_cols:
        all_categories = (
            pd.concat((ds.X_train[col], ds.X_test[col])).fillna(unknown).unique()
        )
        ds.X_train[col] = (
            ds.X_train[col]
            .fillna(unknown)
            .astype("category", categories=all_categories)
        )
        ds.X_test[col] = (
            ds.X_test[col].fillna(unknown).astype("category", categories=all_categories)
        )

    ds.X_train = pd.concat([ds.X_train, pd.get_dummies(ds.X_train[i_cols])], axis=1)
    ds.X_test = pd.concat([ds.X_test, pd.get_dummies(ds.X_test[i_cols])], axis=1)

    ds.X_train = ds.X_train.drop(i_cols, axis=1)
    ds.X_test = ds.X_test.drop(i_cols, axis=1)

    logger.info("Following columns were one hot encoded then dropped")
    logger.info(", ".join(i_cols))


def count_encoding(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    count_together = ["card1", "card2", "card3", "card4", "card5", "card6", "id_36"]
    count_separately = ["id_01", "id_31", "id_33", "id_36"]
    for feature in count_together:
        ds.X_train[feature + "_count_full"] = ds.X_train[feature].map(
            pd.concat(
                [ds.X_train[feature], ds.X_test[feature]], ignore_index=True
            ).value_counts(dropna=False)
        )
        ds.X_test[feature + "_count_full"] = ds.X_test[feature].map(
            pd.concat(
                [ds.X_train[feature], ds.X_test[feature]], ignore_index=True
            ).value_counts(dropna=False)
        )

    # Encoding - count encoding separately for ds.X_train and ds.X_test
    for feature in count_separately:
        ds.X_train[feature + "_count_dist"] = ds.X_train[feature].map(
            ds.X_train[feature].value_counts(dropna=False)
        )
        ds.X_test[feature + "_count_dist"] = ds.X_test[feature].map(
            ds.X_test[feature].value_counts(dropna=False)
        )

    logger.info("Following columns were label encoded, train/test together")
    logger.info(", ".join(count_together))
    logger.info("Following columns were label encoded, train/test separately")
    logger.info(", ".join(count_separately))


def frequency_encoding(ds):
    # https://www.kaggle.com/kyakovlev/ieee-ground-baseline
    i_cols = [
        "card1",
        "card2",
        "card3",
        "card5",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "DeviceInfo_device",
        "DeviceInfo_version",
        "id_30",
        "id_30_device",
        "id_30_version",
        "id_31_device",
        "id_33",
        "uid",
        "uid2",
        "uid3",
    ]

    for col in i_cols:
        temp_df = pd.concat([ds.X_train[[col]], ds.X_test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        ds.X_train[col + "_fq_enc"] = ds.X_train[col].map(fq_encode)
        ds.X_test[col + "_fq_enc"] = ds.X_test[col].map(fq_encode)

    for col in ["DT_M", "DT_W", "DT_D"]:
        temp_df = pd.concat([ds.X_train[[col]], ds.X_test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        ds.X_train[col + "_total"] = ds.X_train[col].map(fq_encode)
        ds.X_test[col + "_total"] = ds.X_test[col].map(fq_encode)

    for period in ["DT_M", "DT_W", "DT_D"]:
        for col in ["uid"]:
            new_column = col + "_" + period

            temp_df = pd.concat([ds.X_train[[col, period]], ds.X_test[[col, period]]])
            temp_df[new_column] = (
                temp_df[col].astype(str) + "_" + (temp_df[period]).astype(str)
            )
            fq_encode = temp_df[new_column].value_counts().to_dict()

            ds.X_train[new_column] = (
                ds.X_train[col].astype(str) + "_" + ds.X_train[period].astype(str)
            ).map(fq_encode)
            ds.X_test[new_column] = (
                ds.X_test[col].astype(str) + "_" + ds.X_test[period].astype(str)
            ).map(fq_encode)

            ds.X_train[new_column] /= ds.X_train[period + "_total"]
            ds.X_test[new_column] /= ds.X_test[period + "_total"]

    logger.info("Columns were frequency encoded")


def label_encoding(ds):
    """
    Apply label encoder to ds.X_train and ds.X_test categorical columns, while preserving nan values

    input: a Dataset
    output: (train, test)
    """
    converted_cols = []
    nan_constant = "unseen_before_label"

    for col in ds.X_train.columns:
        if ds.X_train[col].dtype == object or ds.X_test[col].dtype == object:
            ds.X_train[col] = ds.X_train[col].fillna(nan_constant)
            ds.X_test[col] = ds.X_test[col].fillna(nan_constant)

            lbl = LabelEncoder()
            lbl.fit(list(ds.X_train[col].values) + list(ds.X_test[col].values))
            ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))
            ds.X_test[col] = lbl.transform(list(ds.X_test[col].values))

            if nan_constant in lbl.classes_:
                nan_constant = lbl.transform([nan_constant])[0]
                ds.X_train.loc[ds.X_train[col] == nan_constant, col] = np.nan
                ds.X_test.loc[ds.X_test[col] == nan_constant, col] = np.nan
            converted_cols.append(col)

    logger.info("Following columns were label encoded")
    logger.info(", ".join(converted_cols))


def target_mean_encoding(ds):
    """ProductCD and M4 Target mean
    """
    for col in ["ProductCD", "M4"]:
        temp_dict = (
            ds.X_train.groupby([col])[TARGET]
            .agg(["mean"])
            .reset_index()
            .rename(columns={"mean": col + "_target_mean"})
        )
        temp_dict.index = temp_dict[col].values
        temp_dict = temp_dict[col + "_target_mean"].to_dict()

        ds.X_train[col + "_target_mean"] = ds.X_train[col].map(temp_dict)
        ds.X_test[col + "_target_mean"] = ds.X_test[col].map(temp_dict)


def convert_category_cols_lgb(ds):
    """
    Only necessary for LGB if doing automatic category
    """
    converted_cols = []
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
            converted_cols.append(col)

        ## we also know all categorical columns already xD
        if col in ds.get_categorical_cols():
            ds.X_train[col] = ds.X_train[col].astype("category")
            ds.X_test[col] = ds.X_test[col].astype("category")
            converted_cols.append(col)

    logger.info("Following columns were converted to category")
    logger.info(", ".join(converted_cols))


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

    rm_cols = [
        "TransactionDT",  # These columns are pure noise right now
        "uid",
        "uid2",
        "uid3",  # Our new client uID -> very noisy data
        "DT",
        "DT_M",
        "DT_W",
        "DT_D",  # Temporary Variables
        "DT_hour",
        "DT_day_week",
        "DT_day",
        "DT_D_total",
        "DT_W_total",
        "DT_M_total",
        "id_30",
        "id_31",
        "id_33",
    ]

    cols_to_drop = list(
        set(
            many_null_cols
            + many_null_cols_test
            + big_top_value_cols
            + big_top_value_cols_test
            + one_value_cols
            + one_value_cols_test
            + rm_cols
        )
    )

    ds.X_train = ds.X_train.drop(cols_to_drop, axis=1)
    ds.X_test = ds.X_test.drop(cols_to_drop, axis=1)

    logger.info("Following columns were dropped")
    logger.info(", ".join(cols_to_drop))


def build_processed_dataset(ds):
    clean_inf_nan(ds)
    clean_noise_cards(ds)

    transform_amount(ds)
    parse_emails(ds)
    map_emails(ds)
    parse_device(ds)
    build_transaction_features(ds)
    build_date_features(ds)
    build_interaction_features(ds)
    aggregate_cols(ds)

    # reducer = VestaReducer(ds)
    # reducer.pca(ds)
    # reducer.drop_v_cols(ds)

    binary_encoding(ds)
    # one_hot_encoding(ds)
    count_encoding(ds)
    frequency_encoding(ds)
    label_encoding(ds)
    # target_mean_encoding(ds) # Data leakage, do in each fold separately !

    drop_cols(ds)
