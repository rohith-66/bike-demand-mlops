import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.features import add_time_features, FEATURE_COLS, TARGET_COL

DATA_PATH = Path("data/raw/day.csv")

SEARCH_SPACE = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

def cv_rmse(model, X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    rmses = []
    for tr, va in tscv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmse = np.sqrt(mean_squared_error(y[va], pred))
        rmses.append(rmse)
    return float(np.mean(rmses)), float(np.std(rmses))

def main(n_iter=20, random_state=42):
    print("Loading…")
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df = add_time_features(df).dropna().reset_index(drop=True)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    rng = np.random.RandomState(random_state)
    best = {"score": 1e9, "params": None}

    for i, params in enumerate(ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=rng)):
        model = XGBRegressor(tree_method="hist", random_state=42, **params)
        mean_rmse, std_rmse = cv_rmse(model, X, y)
        print(f"[{i+1}/{n_iter}] {params} -> CV_RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
        if mean_rmse < best["score"]:
            best = {"score": mean_rmse, "params": params}

    print("\nBest params:", best["params"])
    print(f"Best CV RMSE: {best['score']:.2f}")

if __name__ == "__main__":
    main()
