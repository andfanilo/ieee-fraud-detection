import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from path import Path
from src.dataset.utils import df_empty, reduce_mem_usage
from src.utils import get_root_dir

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.submission = None

        self.test_loaded = False

        root_folder = get_root_dir() / "data"
        self.raw_folder = root_folder / "raw"
        self.interim_folder = root_folder / "interim"
        self.processed_folder = root_folder / "processed"
        self.submissions_folder = root_folder / "submissions"

        self.identity_cols = [
            "id_01",
            "id_02",
            "id_03",
            "id_04",
            "id_05",
            "id_06",
            "id_07",
            "id_08",
            "id_09",
            "id_10",
            "id_11",
            "id_12",
            "id_13",
            "id_14",
            "id_15",
            "id_16",
            "id_17",
            "id_18",
            "id_19",
            "id_20",
            "id_21",
            "id_22",
            "id_23",
            "id_24",
            "id_25",
            "id_26",
            "id_27",
            "id_28",
            "id_29",
            "id_30",
            "id_31",
            "id_32",
            "id_33",
            "id_34",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
            "DeviceType",
            "DeviceInfo",
        ]
        self.transaction_cols = [
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
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
            "D9",
            "D10",
            "D11",
            "D12",
            "D13",
            "D14",
            "D15",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "V29",
            "V30",
            "V31",
            "V32",
            "V33",
            "V34",
            "V35",
            "V36",
            "V37",
            "V38",
            "V39",
            "V40",
            "V41",
            "V42",
            "V43",
            "V44",
            "V45",
            "V46",
            "V47",
            "V48",
            "V49",
            "V50",
            "V51",
            "V52",
            "V53",
            "V54",
            "V55",
            "V56",
            "V57",
            "V58",
            "V59",
            "V60",
            "V61",
            "V62",
            "V63",
            "V64",
            "V65",
            "V66",
            "V67",
            "V68",
            "V69",
            "V70",
            "V71",
            "V72",
            "V73",
            "V74",
            "V75",
            "V76",
            "V77",
            "V78",
            "V79",
            "V80",
            "V81",
            "V82",
            "V83",
            "V84",
            "V85",
            "V86",
            "V87",
            "V88",
            "V89",
            "V90",
            "V91",
            "V92",
            "V93",
            "V94",
            "V95",
            "V96",
            "V97",
            "V98",
            "V99",
            "V100",
            "V101",
            "V102",
            "V103",
            "V104",
            "V105",
            "V106",
            "V107",
            "V108",
            "V109",
            "V110",
            "V111",
            "V112",
            "V113",
            "V114",
            "V115",
            "V116",
            "V117",
            "V118",
            "V119",
            "V120",
            "V121",
            "V122",
            "V123",
            "V124",
            "V125",
            "V126",
            "V127",
            "V128",
            "V129",
            "V130",
            "V131",
            "V132",
            "V133",
            "V134",
            "V135",
            "V136",
            "V137",
            "V138",
            "V139",
            "V140",
            "V141",
            "V142",
            "V143",
            "V144",
            "V145",
            "V146",
            "V147",
            "V148",
            "V149",
            "V150",
            "V151",
            "V152",
            "V153",
            "V154",
            "V155",
            "V156",
            "V157",
            "V158",
            "V159",
            "V160",
            "V161",
            "V162",
            "V163",
            "V164",
            "V165",
            "V166",
            "V167",
            "V168",
            "V169",
            "V170",
            "V171",
            "V172",
            "V173",
            "V174",
            "V175",
            "V176",
            "V177",
            "V178",
            "V179",
            "V180",
            "V181",
            "V182",
            "V183",
            "V184",
            "V185",
            "V186",
            "V187",
            "V188",
            "V189",
            "V190",
            "V191",
            "V192",
            "V193",
            "V194",
            "V195",
            "V196",
            "V197",
            "V198",
            "V199",
            "V200",
            "V201",
            "V202",
            "V203",
            "V204",
            "V205",
            "V206",
            "V207",
            "V208",
            "V209",
            "V210",
            "V211",
            "V212",
            "V213",
            "V214",
            "V215",
            "V216",
            "V217",
            "V218",
            "V219",
            "V220",
            "V221",
            "V222",
            "V223",
            "V224",
            "V225",
            "V226",
            "V227",
            "V228",
            "V229",
            "V230",
            "V231",
            "V232",
            "V233",
            "V234",
            "V235",
            "V236",
            "V237",
            "V238",
            "V239",
            "V240",
            "V241",
            "V242",
            "V243",
            "V244",
            "V245",
            "V246",
            "V247",
            "V248",
            "V249",
            "V250",
            "V251",
            "V252",
            "V253",
            "V254",
            "V255",
            "V256",
            "V257",
            "V258",
            "V259",
            "V260",
            "V261",
            "V262",
            "V263",
            "V264",
            "V265",
            "V266",
            "V267",
            "V268",
            "V269",
            "V270",
            "V271",
            "V272",
            "V273",
            "V274",
            "V275",
            "V276",
            "V277",
            "V278",
            "V279",
            "V280",
            "V281",
            "V282",
            "V283",
            "V284",
            "V285",
            "V286",
            "V287",
            "V288",
            "V289",
            "V290",
            "V291",
            "V292",
            "V293",
            "V294",
            "V295",
            "V296",
            "V297",
            "V298",
            "V299",
            "V300",
            "V301",
            "V302",
            "V303",
            "V304",
            "V305",
            "V306",
            "V307",
            "V308",
            "V309",
            "V310",
            "V311",
            "V312",
            "V313",
            "V314",
            "V315",
            "V316",
            "V317",
            "V318",
            "V319",
            "V320",
            "V321",
            "V322",
            "V323",
            "V324",
            "V325",
            "V326",
            "V327",
            "V328",
            "V329",
            "V330",
            "V331",
            "V332",
            "V333",
            "V334",
            "V335",
            "V336",
            "V337",
            "V338",
            "V339",
        ]

        self.identity_cols_categorical = [
            "id_12",
            "id_13",
            "id_14",
            "id_15",
            "id_16",
            "id_17",
            "id_18",
            "id_19",
            "id_20",
            "id_21",
            "id_22",
            "id_23",
            "id_24",
            "id_25",
            "id_26",
            "id_27",
            "id_28",
            "id_29",
            "id_30",
            "id_31",
            "id_32",
            "id_33",
            "id_34",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
            "DeviceType",
            "DeviceInfo",
        ]
        self.transaction_cols_categorical = [
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
        ]
        self.categorical_cols = (
            self.identity_cols_categorical + self.transaction_cols_categorical
        )

    def add_categorical_cols(self, elements):
        """Add custom column to categorical cols
        """
        self.categorical_cols = self.categorical_cols + elements

    def remove_categorical_cols(self, cols_to_drop):
        """Remove custom column from categorical cols
        """
        self.categorical_cols = [
            c for c in self.categorical_cols if c not in cols_to_drop
        ]

    def load_dataset(self, version="", load_test=True):
        self.X_train = pd.read_parquet(
            self.interim_folder / f"X_train_{version}.parquet"
        )
        self.y_train = df_empty(["TransactionID", "isFraud"], dtypes=[str, np.float])
        self.y_train["TransactionID"] = self.X_train.index.to_numpy()
        self.y_train = self.y_train.set_index("TransactionID")
        self.y_train["isFraud"] = np.load(
            self.interim_folder / f"y_train_{version}.npy"
        )

        if load_test:
            self.X_test = pd.read_parquet(
                self.interim_folder / f"X_test_{version}.parquet"
            )
            self.test_loaded = True
        else:
            self.X_test = df_empty(self.X_train.columns, dtypes=self.X_train.dtypes)

        self.submission = self.__build_submission()

    def load_raw(self, nrows=None, load_test=True):
        """
        Load files into Dataset

        nrows: Number of rows to load in training / testing dataset
        load_test: Load test dataset if True. Put to False if you only plan to explore data
        """
        train_transaction = pd.read_csv(
            self.raw_folder / "train_transaction.csv",
            index_col="TransactionID",
            nrows=nrows,
        )

        train_identity = pd.read_csv(
            self.raw_folder / "train_identity.csv", index_col="TransactionID"
        )

        train = train_transaction.merge(
            train_identity,
            how="left",
            left_index=True,
            right_index=True,
            indicator="has_identity",
        )
        train["has_identity"] = train["has_identity"].map({"left_only": 0, "both": 1})

        train = reduce_mem_usage(train)

        self.y_train = train["isFraud"].copy()
        del train_transaction, train_identity

        self.X_train = train.drop("isFraud", axis=1)
        del train

        if load_test:
            test_transaction = pd.read_csv(
                self.raw_folder / "test_transaction.csv",
                index_col="TransactionID",
                nrows=nrows,
            )

            test_identity = pd.read_csv(
                self.raw_folder / "test_identity.csv", index_col="TransactionID"
            )

            test = test_transaction.merge(
                test_identity,
                how="left",
                left_index=True,
                right_index=True,
                indicator="has_identity",
            )
            test["has_identity"] = test["has_identity"].map({"left_only": 0, "both": 1})

            test = reduce_mem_usage(test)
            del test_transaction, test_identity
            self.X_test = test.copy()
            del test
            self.test_loaded = True
        else:
            self.X_test = df_empty(self.X_train.columns, dtypes=self.X_train.dtypes)

        self.submission = self.__build_submission()

    def save_dataset(self, version="", save_test=True):
        self.X_train.to_parquet(
            self.interim_folder / f"X_train_{version}.parquet", index=True
        )
        np.save(self.interim_folder / f"y_train_{version}.npy", self.y_train)
        if save_test:
            self.X_test.to_parquet(
                self.interim_folder / f"X_test_{version}.parquet", index=True
            )

    def write_submission(self, filename):
        self.submission.reset_index().to_csv(
            self.submissions_folder / filename + ".csv", index=False
        )

    def __build_submission(self):
        df = df_empty(["TransactionID", "isFraud"], dtypes=[str, np.float])
        df["TransactionID"] = self.X_test.index.to_numpy()
        df = df.set_index("TransactionID")
        df["isFraud"] = 0.5
        return df
