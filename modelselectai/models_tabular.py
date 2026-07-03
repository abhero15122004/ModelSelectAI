# modelselectai/models_tabular.py
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings

def safe_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def safe_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def safe_import_cat():
    try:
        import catboost as cat
        return cat
    except Exception:
        return None

def classification_models():
    xgb = safe_import_xgb(); lgb = safe_import_lgb(); cat = safe_import_cat()
    models = [
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("decision_tree", DecisionTreeClassifier()),
        ("random_forest", RandomForestClassifier(n_estimators=300)),
        ("svm_rbf", SVC(probability=True)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("mlp", MLPClassifier(max_iter=500)),
    ]
    if xgb:
        models.append(("xgboost", xgb.XGBClassifier(n_estimators=400, eval_metric="logloss", tree_method="hist")))
    if lgb:
        models.append(("lightgbm", lgb.LGBMClassifier(n_estimators=500, verbose=-1)))
    if cat:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models.append(("catboost", cat.CatBoostClassifier(verbose=False, iterations=500)))
    return models

def regression_models():
    xgb = safe_import_xgb(); lgb = safe_import_lgb(); cat = safe_import_cat()
    models = [
        ("linear_regression", LinearRegression()),
        ("ridge", Ridge()),
        ("lasso", Lasso(max_iter=5000)),
        ("decision_tree", DecisionTreeRegressor()),
        ("random_forest", RandomForestRegressor(n_estimators=300)),
        ("svr_rbf", SVR()),
        ("knn", KNeighborsRegressor()),
        ("mlp", MLPRegressor(max_iter=500)),
        ("gbr", GradientBoostingRegressor()),
    ]
    if xgb:
        models.append(("xgboost", xgb.XGBRegressor(n_estimators=500, tree_method="hist")))
    if lgb:
        models.append(("lightgbm", lgb.LGBMRegressor(n_estimators=600, verbose=-1)))
    if cat:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models.append(("catboost", cat.CatBoostRegressor(verbose=False, iterations=600)))
    return models