import numpy as np
import pandas as pd
import umap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class VestaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, V_features):
        """
        Reduce Vesta features
        """
        self.V_features = (
            V_features
        )  # [col for col in ds.X_train.columns.values if col[0] == "V"]
        self.V_groups = self._find_Vgroups(self.V_features)

        self.X_train = ds.X_train[self.V_features].copy().values.astype("float32")
        self.X_test = ds.X_test[self.V_features].copy().values.astype("float32")
        self.y_train = ds.y_train.copy().reset_index()["isFraud"]

        # Mostly count or ranking so can replace NaN by -1
        self.X_train[np.isnan(self.X_train)] = -1
        self.X_test[np.isnan(self.X_test)] = -1

    def fit(self, X, y=None):
        """Fit encoder according to X and y.
        """
        return self

    def transform(self, X, y=None):
        """Perform the transformation to new categorical data.
        """
        return X

    def _find_Vgroups(self, V_variables):
        """Recover groups of Vesta features which are null together
        """
        na_value = X[V_variables].isnull().sum()
        na_list = na_value.unique()
        na_value = na_value.to_dict()
        cols_same_null = []
        for i in range(len(na_list)):
            cols_same_null.append([k for k, v in na_value.items() if v == na_list[i]])
        return cols_same_null

    def pca(self, ds, n_dim=0.9):
        pca = PCA(n_components=n_dim, random_state=42)
        pca.fit(self.X_train)
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)

        for i in range(pca.n_components_):
            col = f"V{i}_pca"
            ds.X_train[col] = X_train_pca[:, i]
            ds.X_test[col] = X_test_pca[:, i]

    def umap(self, ds, n_components=40):
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reducer.fit(self.X_train, y=self.y_train)
        X_train_umap = reducer.transform(self.X_train)
        X_test_umap = reducer.transform(self.X_test)

        for i in range(n_components):
            col = f"V{i}_umap"
            ds.X_train[col] = X_train_umap[:, i]
            ds.X_test[col] = X_test_umap[:, i]

    def drop_v_cols(self, ds):
        """
        Drop original columns at the end
        """
        ds.X_train = ds.X_train.drop(self.V_features, axis=1)
        ds.X_test = ds.X_test.drop(self.V_features, axis=1)
