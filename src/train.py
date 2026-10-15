import mlflow, mlflow.sklearn, numpy as np, pandas as pd, yaml
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clean import clean, FEATURES, FEATURES_NUM, FEATURES_CAT

p = yaml.safe_load(open("params.yaml"))
df = clean(pd.read_csv(p["data_path"]), p)
X, y = df[FEATURES], df[p["target"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=p["test_size"], random_state=p["seed"])

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), FEATURES_NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), FEATURES_CAT),
])

models = {
    "ridge": Ridge(alpha=1.0),
    "rf_100": RandomForestRegressor(n_estimators=100, random_state=p["seed"], n_jobs=-1),
    "rf_300": RandomForestRegressor(n_estimators=300, random_state=p["seed"], n_jobs=-1),
}

mlflow.set_tracking_uri(p["tracking_uri"])
mlflow.set_experiment("house-price")
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = TransformedTargetRegressor(
            regressor=Pipeline([("pre", pre), ("model", model)]),
            func=np.log1p, inverse_func=np.expm1).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        mlflow.log_params({"model": name, "n_train": len(X_tr), **model.get_params()})
        mlflow.log_metrics({
            "mae": mean_absolute_error(y_te, pred),
            "rmse": root_mean_squared_error(y_te, pred),
            "r2": r2_score(y_te, pred)})
        mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                 serialization_format="cloudpickle")
        print(name, "r2", round(r2_score(y_te, pred), 3), "mae", round(mean_absolute_error(y_te, pred)))