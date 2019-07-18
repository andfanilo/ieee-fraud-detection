import xgboost as xgb
import lightgbm as lgb


def train_xgb(ds):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        n_jobs=4,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        missing=-999
    )

    clf.fit(ds.X_train, ds.y_train)
    return clf


def train_lgb(ds):
    clf = lgb.LGBMClassifier(
        max_depth=9,
        random_state=314,
        silent=True,
        metric="None",
        n_jobs=4,
        n_estimators=500,
        colsample_bytree=0.9,
        subsample=0.9,
        learning_rate=0.1,
    )

    clf.fit(ds.X_train, ds.y_train)
    return clf
