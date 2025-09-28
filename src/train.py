import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost.callback import EarlyStopping

from src.features import add_time_features, FEATURE_COLS, TARGET_COL


DATA_PATH = Path("data/raw/day.csv")
MODEL_PATH = Path("models/bike_xgb_model.json")  # save as JSON (native format)
MODEL_PATH.parent.mkdir(exist_ok=True)


TARGET_COL = "cnt"

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape}

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df = add_time_features(df).dropna().reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # time-based split
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # early stopping validation split (from training chunk)
    es_idx = int(len(X_train) * 0.85)
    X_tr, X_es = X_train[:es_idx], X_train[es_idx:]
    y_tr, y_es = y_train[:es_idx], y_train[es_idx:]

    # wrap in DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
    desv = xgb.DMatrix(X_es, label=y_es, feature_names=FEATURE_COLS)
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)

    # parameters
    params = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.9,
        "lambda": 1.5,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42
    }

    print("Training with early stopping...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(desv, "validation")],
        callbacks=[EarlyStopping(rounds=50)]
    )

    # evaluate on true test set
    y_pred = model.predict(dtest)
    metrics = regression_metrics(y_test, y_pred)
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.2f}")

    # save model
    model.save_model(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()
