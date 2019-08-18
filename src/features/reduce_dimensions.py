import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class VestaReducer:
    def __init__(self, ds):
        """
        Reduce Vesta features
        """
        self.V_features = [col for col in ds.X_train.columns.values if col[0] == "V"]

        self.X_train = ds.X_train[self.V_features].copy().values.astype("float32")
        self.X_test = ds.X_test[self.V_features].copy().values.astype("float32")
        self.y_train = ds.y_train.copy().reset_index()["isFraud"]

        self.X_train[np.isnan(self.X_train)] = 0
        self.X_test[np.isnan(self.X_test)] = 0

        scaler = StandardScaler().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

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
