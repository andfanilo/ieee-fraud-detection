import numpy as np
import pandas as pd
import matplotlib.pylab as pl

import shap

shap.initjs()


class TreeVisualizer:
    def __init__(self, clf, X, interactions=False):
        """
        interactions takes a couple minutes since SHAP interaction values take a factor of 2 * # features
        more time than SHAP values to compute
        """
        self.X = X
        self.interactions = False
        self.explainer = shap.TreeExplainer(clf)
        self.shap_values = self.explainer.shap_values(self.X)

        if interactions:
            self.interactions=True
            self.shap_interaction_values = self.explainer.shap_interaction_values(self.X)

    def get_importances(self):
        """
        from https://www.kaggle.com/wrosinski/shap-feature-importance-with-feature-engineering
        """
        shap_sum = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame([self.X.columns.tolist(), shap_sum.tolist()]).T
        importance_df.columns = ['column_name', 'shap_importance']
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        return importance_df

    def get_interaction_importances(self):
        """
        from https://www.kaggle.com/wrosinski/shap-feature-importance-with-feature-engineering
        """
        return np.abs(self.shap_interaction_values).mean(axis=0)

    def interaction_matrix(self):
        """
        from https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
        """
        tmp = np.abs(self.shap_interaction_values).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i,i] = 0
        inds = np.argsort(-tmp.sum(0))[:50]
        tmp2 = tmp[inds,:][:,inds]
        pl.figure(figsize=(12,12))
        pl.imshow(tmp2)
        pl.yticks(range(tmp2.shape[0]), self.X.columns[inds], rotation=50.4, horizontalalignment="right")
        pl.xticks(range(tmp2.shape[0]), self.X.columns[inds], rotation=50.4, horizontalalignment="left")
        pl.gca().xaxis.tick_top()
        pl.show()

    def summary_bar_plot(self):
        return shap.summary_plot(self.shap_values, self.X, plot_type="bar")

    def summary_plot(self):
        return shap.summary_plot(self.shap_values, self.X)

    def interaction_summary_plot(self):
        """
        A summary plot of a SHAP interaction value matrix plots a matrix of summary plots 
        with the main effects on the diagonal and the interaction effects off the diagonal.
        """
        if not self.interactions:
            return
        return shap.summary_plot(self.shap_interaction_values, self.X)

    def dependence_plot(self, feature):
        return shap.dependence_plot(feature, self.shap_values, self.X)

    def interaction_dependence_plot(self, feature):
        """
        Takes tuple (feature1, feature2) as input
        """
        if not self.interactions:
            return
        return shap.dependence_plot(feature, self.shap_interaction_values, self.X)

    def force_plot(self):
        return shap.force_plot(self.explainer.expected_value, self.shap_values, self.X)
