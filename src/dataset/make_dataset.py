from path import Path

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


class Dataset:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None

        root_folder = Path("../../data")
        self.raw_folder = root_folder / "raw"
        self.interim_folder = root_folder / "interim"
        self.processed_folder = root_folder / "processed"
        self.submissions_folder = root_folder / "submissions"

    def load_dataset(self):
        self.X_train = pd.read_parquet(self.interim_folder / 'X_train.parquet')
        self.X_test = pd.read_parquet(self.interim_folder / 'X_train.parquet')
        self.y_train = np.load(self.interim_folder / 'y_train.npy')

    def load_raw(self, nrows=None):
        train_transaction = pd.read_csv(
            self.raw_folder / "train_transaction.csv", index_col="TransactionID", nrows=nrows
        )
        test_transaction = pd.read_csv(
            self.raw_folder / "test_transaction.csv", index_col="TransactionID", nrows=nrows
        )

        train_identity = pd.read_csv(
            self.raw_folder / "train_identity.csv", index_col="TransactionID"
        )
        test_identity = pd.read_csv(
            self.raw_folder / "test_identity.csv", index_col="TransactionID"
        )

        train = train_transaction.merge(
            train_identity, how="left", left_index=True, right_index=True
        )
        test = test_transaction.merge(
            test_identity, how="left", left_index=True, right_index=True
        )

        self.sample_submission = pd.read_csv(
            self.submissions_folder / "sample_submission.csv", index_col="TransactionID"
        )

        train = self.__reduce_mem_usage(train)
        test = self.__reduce_mem_usage(test)

        self.y_train = train["isFraud"].copy()
        del train_transaction, train_identity, test_transaction, test_identity


        self.X_train = train.drop("isFraud", axis=1)
        self.X_test = test.copy()
        del train, test

    def get_features(self):
        """Get a copy of X_train, X_test"""
        return (self.X_train.copy(), self.X_test.copy())

    def get_labels(self):
        """Get a copy of y_train"""
        return self.y_train.copy()

    def save_dataset(self):
        self.X_train.to_parquet(self.interim_folder / 'X_train.parquet')
        self.X_test.to_parquet(self.interim_folder / 'X_test.parquet')
        np.save(self.interim_folder / 'y_train.npy', self.y_train)

    def __reduce_mem_usage(self, df):
        """
        From https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee
        Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        Reducing Memory Size of dataset
        """
        start_mem_usg = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        NAlist = []  # Keeps track of columns that have missing values filled in.
        for col in df.columns:
            if df[col].dtype != object:  # Exclude strings
                # Print current column type
                #print("******************************")
                #print("Column: ", col)
                #print("dtype before: ", df[col].dtype)
                # make variables for Int, max and min
                IsInt = False
                mx = df[col].max()
                mn = df[col].min()
                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(df[col]).all():
                    NAlist.append(col)
                    df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
                asint = df[col].fillna(0).astype(np.int64)
                result = df[col] - asint
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif (
                            mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max
                        ):
                            df[col] = df[col].astype(np.int16)
                        elif (
                            mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max
                        ):
                            df[col] = df[col].astype(np.int32)
                        elif (
                            mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max
                        ):
                            df[col] = df[col].astype(np.int64)
                # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)

                # Print new column type
                #print("dtype after: ", df[col].dtype)
                #print("******************************")
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
        print("_________________")
        print("")
        print(
            "Warning: the following columns have missing values filled with 'df['column_name'].min() -1': "
        )
        print("_________________")
        print("")
        print(NAlist)
        print("")
        return df

