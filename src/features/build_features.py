import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.features.utils import num_after_point

logger = logging.getLogger(__name__)


def clean_inf_nan(ds):
    # by https://www.kaggle.com/dimartinot
    ds.X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("Infinites were cleaned")


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

    logger.info(
        f"Noise card1 (less than {valid_counts} counts in training) were dropped"
    )


def id_31_check_latest_browser(ds):
    for df in [ds.X_train, ds.X_test]:
        # check latest browser with id_31
        # https://www.kaggle.com/amirhmi/a-complete-feature-engineering-with-lgbm
        df["lastest_browser"] = np.zeros(df.shape[0])
        df.loc[df["id_31"] == "samsung browser 7.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "opera 53.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "mobile safari 10.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "google search application 49.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "firefox 60.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "edge 17.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 69.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 67.0 for android", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 63.0 for android", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 63.0 for ios", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 64.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 64.0 for android", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 64.0 for ios", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 65.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 65.0 for android", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 65.0 for ios", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 66.0", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 66.0 for android", "lastest_browser"] = 1
        df.loc[df["id_31"] == "chrome 66.0 for ios", "lastest_browser"] = 1


def clean_id_31(ds):
    for df in [ds.X_train, ds.X_test]:
        # Extract browser and version beforehand
        # df["id_31_browser"] = df["id_31"].str.split(" ", expand=True)[0]
        # df["id_31_version"] = df["id_31"].str.split(" ", expand=True)[1]

        # Clean id_31, this stays categorical
        df["id_31"] = df["id_31"].str.replace("([0-9\.])", "")
        df["id_31"][df["id_31"].str.contains("chrome", regex=False) == True] = "chrome"
        df["id_31"][
            df["id_31"].str.contains("Samsung", regex=False) == True
        ] = "samsung"
        df["id_31"][
            df["id_31"].str.contains("samsung", regex=False) == True
        ] = "samsung"
        df["id_31"][
            df["id_31"].str.contains("firefox", regex=False) == True
        ] = "firefox"
        df["id_31"][df["id_31"].str.contains("safari", regex=False) == True] = "safari"
        df["id_31"][df["id_31"].str.contains("opera", regex=False) == True] = "opera"
        df["id_31"] = df["id_31"].str.replace(" ", "")
        df.loc[
            df["id_31"].str.contains("Generic/Android", na=False), "id_31"
        ] = "android"
        df.loc[
            df["id_31"].str.contains("androidbrowser", na=False), "id_31"
        ] = "android"
        df.loc[
            df["id_31"].str.contains("androidwebview", na=False), "id_31"
        ] = "android"
        df.loc[df["id_31"].str.contains("android", na=False), "id_31"] = "android"
        df.loc[df["id_31"].str.contains("chromium", na=False), "id_31"] = "chrome"
        df.loc[df["id_31"].str.contains("google", na=False), "id_31"] = "chrome"
        df.loc[
            df["id_31"].str.contains("googlesearchapplication", na=False), "id_31"
        ] = "chrome"
        df.loc[df["id_31"].str.contains("iefordesktop", na=False), "id_31"] = "ie"
        df.loc[df["id_31"].str.contains("iefortablet", na=False), "id_31"] = "ie"
        df.loc[
            df.id_31.isin(df.id_31.value_counts()[df.id_31.value_counts() < 20].index),
            "id_31",
        ] = "rare"

        ########################### Browser
        df["id_31"] = df["id_31"].fillna("unknown_device").str.lower()
        df["id_31_device"] = df["id_31"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )

    ds.add_categorical_cols(
        [
            # "id_31_browser",
            # "id_31_version",
            "id_31_device"
        ]
    )

    logger.info("id_31 were processed")


def parse_id_30(ds):
    for df in [ds.X_train, ds.X_test]:
        df["id_30_OS"] = df["id_30"].str.split(" ", expand=True)[0]
        df["id_30_version"] = df["id_30"].str.split(" ", expand=True)[1]

        df["is_win8_vista"] = (df.id_30_OS == "Windows") & (
            (df.id_30_version == "8") | (df.id_30_version == "Vista")
        )
        df["is_win8_vista"] = df["is_win8_vista"].astype(int)
        df["is_windows_otheros"] = (df.DeviceInfo == "Windows") & (
            (df.id_30_OS == "Linux") | (df.id_30_OS == "other")
        )
        df["is_windows_otheros"] = df["is_windows_otheros"].astype(int)

        df.drop("id_30_OS", axis=1, inplace=True)
        df.drop("id_30_version", axis=1, inplace=True)

        df["id_30"] = df["id_30"].fillna("unknown_device").str.lower()
        df["id_30_os"] = df["id_30"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )
        df["id_30_version"] = df["id_30"].apply(
            lambda x: "".join([i for i in x if i.isnumeric()])
        )

    ds.add_categorical_cols(
        ["is_win8_vista", "is_windows_otheros", "id_30_os", "id_30_version"]
    )

    logger.info("Parsed id_30")


def parse_device_info(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    for df in [ds.X_train, ds.X_test]:
        df["device_name"] = df["DeviceInfo"].str.split("/", expand=True)[0]
        df["device_version"] = df["DeviceInfo"].str.split("/", expand=True)[1]

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
        ] = "rare"

        df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown_device").str.lower()
        df["deviceInfo_device"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )
        df["deviceInfo_version"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isnumeric()])
        )

    ds.add_categorical_cols(
        ["device_name", "device_version", "deviceInfo_device", "deviceInfo_version"]
    )

    logger.info("DeviceInfo were parsed")


def id_33_extract_screen_resolution(ds):
    for df in [ds.X_train, ds.X_test]:
        df["screen_width"] = (
            df["id_33"].str.split("x", expand=True)[0].fillna(0).astype(int)
        )
        df["screen_height"] = (
            df["id_33"].str.split("x", expand=True)[1].fillna(0).astype(int)
        )
        df["screen_resolution"] = np.sqrt(df["screen_width"] * df["screen_height"])

    logger.info("Screen resolution extracted")


def transform_amount(ds):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    for df in [ds.X_train, ds.X_test]:
        # decimal part of the transaction amount.
        df["TransactionAmt_main"] = np.floor(df["TransactionAmt"])
        df["TransactionAmt_cents"] = df["TransactionAmt"] - np.floor(
            df["TransactionAmt"]
        )

        # Capture number of decimals
        df["TransactionAmt_n_decimals"] = df["TransactionAmt"].apply(num_after_point)

        # Capture 0, 50, 95 cents of ProductCD W
        df["W_0_cents"] = np.zeros(df.shape[0])
        df["W_50_cents"] = np.zeros(df.shape[0])
        df["W_95_cents"] = np.zeros(df.shape[0])
        df.loc[
            (df["ProductCD"] == "W") & (np.round(df["TransactionAmt_cents"], 2) == 0),
            "W_0_cents",
        ] = 1
        df.loc[
            (df["ProductCD"] == "W") & (np.round(df["TransactionAmt_cents"], 2) == 0.5),
            "W_50_cents",
        ] = 1
        df.loc[
            (df["ProductCD"] == "W")
            & (np.round(df["TransactionAmt_cents"], 2) == 0.95),
            "W_95_cents",
        ] = 1

        # log of transaction amount.
        df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])

    # Check if the Transaction Amount is common or not (we can use freq encoding here)
    # In our dialog with a model we are telling to trust or not to these values
    # ds.X_train["TransactionAmt_check"] = np.where(
    #    ds.X_train["TransactionAmt"].isin(ds.X_test["TransactionAmt"]), 1, 0
    # )
    # ds.X_test["TransactionAmt_check"] = np.where(
    #    ds.X_test["TransactionAmt"].isin(ds.X_train["TransactionAmt"]), 1, 0
    # )

    logger.info("TransactionAmt was worked heavily")


def transform_productCD(ds):
    for df in [ds.X_train, ds.X_test]:
        df["ProductCD"] = df["ProductCD"].map({"W": 0, "C": 1, "R": 2, "H": 3, "S": 4})

    logger.info("ProductCD was label encoded")


def transform_emails_and_domains(ds):
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

    purchaser = "P_emaildomain"
    recipient = "R_emaildomain"
    unknown = "email_not_provided"

    for df in [ds.X_train, ds.X_test]:
        df["is_proton_mail"] = (df[purchaser] == "protonmail.com") | (
            df[recipient] == "protonmail.com"
        )
        df["email_check"] = np.where(
            (df[purchaser] == df[recipient]) & (df[purchaser] != unknown), 1, 0
        )

        for c in [purchaser, recipient]:
            df[purchaser] = df[purchaser].fillna(unknown)
            df[recipient] = df[recipient].fillna(unknown)

            df[c + "_bin"] = df[c].map(emails)

            df[c + "_suffix"] = df[c].map(lambda x: str(x).split(".")[-1])

            df[c + "_suffix"] = df[c + "_suffix"].map(
                lambda x: x if str(x) not in us_emails else "us"
            )

    ds.add_categorical_cols(
        [
            "P_emaildomain_bin",
            "P_emaildomain_suffix",
            "R_emaildomain_bin",
            "R_emaildomain_suffix",
        ]
    )

    logger.info("Emails were transformed")


def encode_M_variables(ds):
    """Encode all M columns except M4
    """
    i_cols = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]

    for df in [ds.X_train, ds.X_test]:
        df["M_sum"] = df[i_cols].sum(axis=1).astype(np.int8)
        df["M_na"] = df[i_cols].isna().sum(axis=1).astype(np.int8)
        for col in i_cols:
            df[col] = df[col].map({"T": 1, "F": 0})

    logger.info("Following columns were binary encoded")
    logger.info(", ".join(i_cols))

    for df in [ds.X_train, ds.X_test]:
        df["M4"] = df["M4"].map({"M0": 1, "M1": 2, "M2": 3})

    logger.info("M4 was label encoded")


def build_uid(ds):
    # Let's add some kind of client uID based on cardID and addr columns
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
        df["uid3"] = df["uid2"].astype(str) + "_" + df["addr1"].astype(str)
        df["uid4"] = df["uid3"].astype(str) + "_" + df["P_emaildomain"].astype(str)

    logger.info("UUIDs were built")


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

    logger.info("Built date features")


def aggregate(ds):
    # For our model current TransactionAmt is a noise
    # https://www.kaggle.com/kyakovlev/ieee-check-noise
    # (even if features importances are telling contrariwise)
    # There are many unique values and model doesn't generalize well
    # Lets do some aggregations
    i_cols = ["card1", "card2", "card3", "card5", "uid", "uid2", "uid3", "uid4"]

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

    ds.add_categorical_cols(random_intersection + my_intersections)

    logger.info("Following columns were created for interaction")
    logger.info(", ".join(random_intersection + my_intersections))


def label_encoding(ds):
    """
    Apply label encoder to ds.X_train and ds.X_test categorical columns, while preserving nan values

    input: a Dataset
    output: (train, test)
    """
    converted_cols = []
    nan_constant = -999

    for col in ds.categorical_cols:
        ds.X_train[col] = ds.X_train[col].fillna(nan_constant)
        ds.X_test[col] = ds.X_test[col].fillna(nan_constant)

        lbl = LabelEncoder()
        lbl.fit(list(ds.X_train[col].values) + list(ds.X_test[col].values))
        ds.X_train[col] = lbl.transform(list(ds.X_train[col].values))
        ds.X_test[col] = lbl.transform(list(ds.X_test[col].values))

        if nan_constant in lbl.classes_:
            nan_transformed = lbl.transform([nan_constant])[0]
            ds.X_train.loc[ds.X_train[col] == nan_transformed, col] = np.nan
            ds.X_test.loc[ds.X_test[col] == nan_transformed, col] = np.nan
        converted_cols.append(col)

    logger.info("Following columns were label encoded")
    logger.info(", ".join(converted_cols))


def build_processed_dataset(ds):
    clean_inf_nan(ds)  # 0.910084 - baseline

    ########################### IDENTITY

    # --- Frankly except maybe for aggregating, those columns don't add a lot of value

    clean_noise_cards(ds)  # from 0.917186 to 0.918363
    id_31_check_latest_browser(ds)  # from 0.910084 to 0.907111
    id_33_extract_screen_resolution(ds)  # from 0.9175 to 0.913483

    # the following 3 gets us from 0.917509 to 0.917186
    clean_id_31(ds)  #  from 0.917509 to 0.914228
    parse_id_30(ds)  #  from 0.917509 to 0.914228
    parse_device_info(ds)  #  from 0.917509 to 0.915502

    # We get to 0.9182 with all activated

    ########################### TRANSACTIONS

    transform_amount(ds)  # down from 0.9182 to 0.9167
    transform_productCD(ds)
    transform_emails_and_domains(ds)
    encode_M_variables(ds)

    ########################### AGGREGATING

    build_uid(ds)
    build_date_features(ds)

    aggregate(ds)
    build_interaction_features(ds)

    ########################### ENCODING

    label_encoding(ds)
