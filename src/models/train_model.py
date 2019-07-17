import xgboost as xgb
import lightgbm as lgb

def train_xgb(ds):
    clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.7,
                        colsample_bytree=0.7,
                        missing=-999,
                        gamma=0.1)

    clf.fit(ds.X_train, ds.y_train)
    return clf