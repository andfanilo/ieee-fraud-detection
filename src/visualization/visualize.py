import shap

shap.initjs()

def summary_plot(ds, clf):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(ds.X_train)
    return shap.summary_plot(shap_values, ds.X_train, plot_type="bar")
