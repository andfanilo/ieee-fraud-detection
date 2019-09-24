import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.features.vesta_transformer import VestaTransformer

logger = logging.getLogger(__name__)


########################### TRANSFORMING


def impute_values(fold):
    for col in fold.categorical_cols:
        fold.X_train[col] = fold.X_train[col].fillna(-999)
        fold.X_valid[col] = fold.X_valid[col].fillna(-999)
        fold.X_test[col] = fold.X_test[col].fillna(-999)

    logger.info("Imputed missing values")


def reduce_V_features(fold):
    vt = VestaTransformer()
    vt.fit(fold.X_train)
    fold.X_train = vt.transform(fold.X_train)
    fold.X_valid = vt.transform(fold.X_valid)
    fold.X_test = vt.transform(fold.X_test)

    logger.info("Reduced V features")


########################### ENCODING


def one_hot_encoding(fold):
    # https://markhneedham.com/blog/2017/07/05/pandasscikit-learn-get_dummies-testtrain-sets-valueerror-shapes-not-aligned/
    i_cols = []
    unknown = "?"

    for col in i_cols:
        all_categories = (
            pd.concat((fold.X_train[col], fold.X_valid[col], fold.X_test[col]))
            .fillna(unknown)
            .unique()
        )
        fold.X_train[col] = (
            fold.X_train[col]
            .fillna(unknown)
            .astype("category", categories=all_categories)
        )
        fold.X_valid[col] = (
            fold.X_valid[col]
            .fillna(unknown)
            .astype("category", categories=all_categories)
        )
        fold.X_test[col] = (
            fold.X_test[col]
            .fillna(unknown)
            .astype("category", categories=all_categories)
        )

    fold.X_train = pd.concat(
        [fold.X_train, pd.get_dummies(fold.X_train[i_cols])], axis=1
    )
    fold.X_valid = pd.concat(
        [fold.X_valid, pd.get_dummies(fold.X_valid[i_cols])], axis=1
    )
    fold.X_test = pd.concat([fold.X_test, pd.get_dummies(fold.X_test[i_cols])], axis=1)

    fold.X_train.drop(i_cols, axis=1, inplace=True)
    fold.X_valid.drop(i_cols, axis=1, inplace=True)
    fold.X_test.drop(i_cols, axis=1, inplace=True)

    logger.info("Following columns were one hot encoded then dropped")
    logger.info(", ".join(i_cols))


def count_encoding(fold):
    """
    https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
    """
    count_together = ["card1", "card2", "card3", "card4", "card5", "card6", "id_36"]
    count_separately = ["id_01", "id_31", "id_33", "id_36"]
    for feature in count_together:
        feature_count = pd.concat(
            [fold.X_train[feature], fold.X_valid[feature]], ignore_index=True
        ).value_counts(dropna=False)

        fold.X_train[feature + "_count_full"] = fold.X_train[feature].map(feature_count)
        fold.X_valid[feature + "_count_full"] = fold.X_valid[feature].map(feature_count)
        fold.X_test[feature + "_count_full"] = fold.X_test[feature].map(feature_count)

    for feature in count_separately:
        # TODO : I think this should be on rolling window instead, count on non length similar fold is dubious
        fold.X_train[feature + "_count_dist"] = fold.X_train[feature].map(
            fold.X_train[feature].value_counts(dropna=False)
        )
        fold.X_valid[feature + "_count_dist"] = fold.X_valid[feature].map(
            fold.X_valid[feature].value_counts(dropna=False)
        )
        fold.X_test[feature + "_count_dist"] = fold.X_test[feature].map(
            fold.X_test[feature].value_counts(dropna=False)
        )

    logger.info("Count encoding done")


def frequency_encoding(fold):
    # https://www.kaggle.com/kyakovlev/ieee-ground-baseline
    i_cols = [
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "ProductCD",
        "product_type",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "deviceInfo_device",
        "deviceInfo_version",
        "id_30",
        "id_30_os",
        "id_30_version",
        "id_31_device",
        "id_33",
        "uid",
        "uid2",
        "uid3",
        "uid4",
        "uid5",
        "bank_type",
    ]

    for col in i_cols:
        temp_df = pd.concat(
            [fold.X_train[[col]], fold.X_valid[[col]], fold.X_test[[col]]]
        )
        fq_encode = temp_df[col].value_counts().to_dict()
        fold.X_train[col + "_fq_enc"] = fold.X_train[col].map(fq_encode)
        fold.X_valid[col + "_fq_enc"] = fold.X_valid[col].map(fq_encode)
        fold.X_test[col + "_fq_enc"] = fold.X_test[col].map(fq_encode)

    for col in ["DT_M", "DT_W", "DT_D", "local_time_M", "local_time_W", "local_time_D"]:
        temp_df = pd.concat(
            [fold.X_train[[col]], fold.X_valid[[col]], fold.X_test[[col]]]
        )
        fq_encode = temp_df[col].value_counts().to_dict()
        fold.X_train[col + "_total"] = fold.X_train[col].map(fq_encode)
        fold.X_valid[col + "_total"] = fold.X_valid[col].map(fq_encode)
        fold.X_test[col + "_total"] = fold.X_test[col].map(fq_encode)

    # timeblock frequency encoding
    for period in [
        "DT_M",
        "DT_W",
        "DT_D",
        "local_time_M",
        "local_time_W",
        "local_time_D",
    ]:
        for col in ["uid", "uid2", "uid3", "uid4", "uid5", "bank_type", "product_type"]:
            new_column = col + "_" + period

            temp_df = pd.concat(
                [
                    fold.X_train[[col, period]],
                    fold.X_valid[[col, period]],
                    fold.X_test[[col, period]],
                ]
            )
            temp_df[new_column] = (
                temp_df[col].astype(str) + "_" + (temp_df[period]).astype(str)
            )
            fq_encode = temp_df[new_column].value_counts().to_dict()

            fold.X_train[new_column] = (
                fold.X_train[col].astype(str) + "_" + fold.X_train[period].astype(str)
            ).map(fq_encode)
            fold.X_valid[new_column] = (
                fold.X_valid[col].astype(str) + "_" + fold.X_valid[period].astype(str)
            ).map(fq_encode)
            fold.X_test[new_column] = (
                fold.X_test[col].astype(str) + "_" + fold.X_test[period].astype(str)
            ).map(fq_encode)

            fold.X_train[new_column] = (
                fold.X_train[new_column] / fold.X_train[period + "_total"]
            )
            fold.X_valid[new_column] = (
                fold.X_valid[new_column] / fold.X_valid[period + "_total"]
            )
            fold.X_test[new_column] = (
                fold.X_test[new_column] / fold.X_test[period + "_total"]
            )

            # Boruta destroyed those so removed for now
            # fold.X_train[f"{new_column}_proportions"] = (
            #    fold.X_train[period + "_total"] / fold.X_train[period + "_total"]
            # )
            # fold.X_valid[f"{new_column}_proportions"] = (
            #    fold.X_valid[period + "_total"] / fold.X_valid[period + "_total"]
            # )
            # fold.X_test[f"{new_column}_proportions"] = (
            #    fold.X_test[period + "_total"] / fold.X_test[period + "_total"]
            # )

    logger.info("Following columns were frequency encoded")
    logger.info(", ".join(i_cols))


def label_encoding(fold):
    """
    Apply label encoder to fold.X_train and fold.X_test categorical columns, while preserving nan values

    input: a Dataset
    output: (train, test)
    """
    converted_cols = []
    nan_constant = -999

    for col in fold.categorical_cols:
        fold.X_train[col] = fold.X_train[col].fillna(nan_constant)
        fold.X_valid[col] = fold.X_valid[col].fillna(nan_constant)
        fold.X_test[col] = fold.X_test[col].fillna(nan_constant)

        lbl = LabelEncoder()
        lbl.fit(
            list(fold.X_train[col].values)
            + list(fold.X_valid[col].values)
            + list(fold.X_test[col].values)
        )
        fold.X_train[col] = lbl.transform(list(fold.X_train[col].values))
        fold.X_valid[col] = lbl.transform(list(fold.X_valid[col].values))
        fold.X_test[col] = lbl.transform(list(fold.X_test[col].values))

        if nan_constant in lbl.classes_:
            nan_transformed = lbl.transform([nan_constant])[0]
            fold.X_train.loc[fold.X_train[col] == nan_transformed, col] = np.nan
            fold.X_valid.loc[fold.X_valid[col] == nan_transformed, col] = np.nan
            fold.X_test.loc[fold.X_test[col] == nan_transformed, col] = np.nan
        converted_cols.append(col)

    logger.info("Following columns were label encoded")
    logger.info(", ".join(converted_cols))


########################### DROPPING


def drop_user_generated_cols(fold):
    rm_cols = [
        "uid",
        "uid2",
        "uid3",
        "uid4",
        "uid5",
        "bank_type",
        "DT",
        "DT_M",
        "DT_W",
        "DT_D",
        "local_time",
        "local_time_M",
        "local_time_W",
        "local_time_D",
        # "DT_M_total",
        # "DT_W_total",
        # "DT_D_total",
        # "DT_hour",
        # "DT_day_week",
        # "DT_day_month",
    ]
    for df in [fold.X_train, fold.X_valid, fold.X_test]:
        df.drop(rm_cols, axis=1, inplace=True)

    fold.remove_categorical_cols(rm_cols)

    logger.info("Temporary columns were dropped")
    logger.info(", ".join(rm_cols))


def drop_cols_auto(fold):
    one_value_cols = [
        col for col in fold.X_train.columns if fold.X_train[col].nunique() <= 1
    ]
    one_value_cols_test = [
        col for col in fold.X_test.columns if fold.X_test[col].nunique() <= 1
    ]

    many_null_cols = [
        col
        for col in fold.X_train.columns
        if fold.X_train[col].isnull().sum() / fold.X_train.shape[0] > 0.9
    ]
    many_null_cols_test = [
        col
        for col in fold.X_test.columns
        if fold.X_test[col].isnull().sum() / fold.X_test.shape[0] > 0.9
    ]

    big_top_value_cols = [
        col
        for col in fold.X_train.columns
        if fold.X_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]
    big_top_value_cols_test = [
        col
        for col in fold.X_test.columns
        if fold.X_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]

    # for col in fold.categorical_cols:
    #    num_unique_values = fold.X_train[col].append(fold.X_test[col]).nunique()
    #    num_total_values = fold.X_train[col].shape[0] + fold.X_test[col].shape[0]

    #    # if num_unique_values / num_total_values > 0.5:
    #    # should do some binning I think
    #    # remove_cat_col.append(col)

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

    # for df in [fold.X_train, fold.X_valid, fold.X_test]:
    #    df.drop(cols_to_drop, axis=1, inplace=True)

    # fold.remove_categorical_cols(cols_to_drop)

    logger.info("Following columns would be dropped because many nulls")
    logger.info(", ".join(list(set(many_null_cols + many_null_cols_test))))
    logger.info("Following columns would be dropped because big top values")
    logger.info(", ".join(list(set(big_top_value_cols + big_top_value_cols_test))))
    logger.info("Following columns would be dropped because one value cols")
    logger.info(", ".join(list(set(one_value_cols + one_value_cols_test))))


def drop_cols_manual(fold):
    rm_cols = ["TransactionDT", "id_30", "id_31", "id_33", "DeviceInfo"]
    for df in [fold.X_train, fold.X_valid, fold.X_test]:
        df.drop(rm_cols, axis=1, inplace=True)

    fold.remove_categorical_cols(rm_cols)

    logger.info("Manual columns were dropped")
    logger.info(", ".join(rm_cols))


def process_fold(fold):
    label_encoding(fold)

    ########################### TRANSFORMING
    impute_values(fold)
    reduce_V_features(fold)

    ########################### ENCODING

    count_encoding(fold)
    frequency_encoding(fold)

    ########################### DROPPING

    drop_user_generated_cols(fold)
    drop_cols_auto(fold)
    drop_cols_manual(fold)

    logger.info("Fold was preprocessed correctly")
