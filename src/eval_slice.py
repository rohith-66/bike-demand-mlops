from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
from src.features import add_time_features

MODEL_PATH = Path("models/bike_xgb_model.json")
DATA_PATH = Path("data/raw/day.csv")

def load_model():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100)
    return rmse, mae, mape

if __name__ == "__main__":
    # Defaults
    start = "2012-07-01"
    end = "2012-07-31"

    # Allow overriding from command line
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]

    print(f"Evaluating slice {start} → {end}")

    model = load_model()
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df = df.sort_values("dteday").reset_index(drop=True)

    # Build features across full dataset
    feats = add_time_features(df).dropna().reset_index(drop=True)

    preds, actuals, dates = [], [], []
    expected = model.feature_names

    for i in range(len(feats)):
        row = feats.iloc[[i]].copy()
        d = row["dteday"].iloc[0]
        if not (pd.to_datetime(start) <= d <= pd.to_datetime(end)):
            continue

        # Ensure all expected features exist
        for col in expected:
            if col not in row:
                row[col] = 0.0
        row = row[expected]

        y = df.loc[df["dteday"] == d, "cnt"].iloc[0]
        dm = xgb.DMatrix(row.values, feature_names=expected)
        yhat = float(model.predict(dm)[0])

        dates.append(d)
        actuals.append(y)
        preds.append(yhat)

    # Metrics
    rmse, mae, mape = metrics(actuals, preds)
    print(f"Slice {start}..{end} -> RMSE: {rmse:.2f}  MAE: {mae:.2f}  MAPE: {mape:.2f}")

    # Show sample rows
    for d, a, p in list(zip(dates, actuals, preds))[:5]:
        print(f"{d.date()}  actual={a:.0f}  pred={p:.0f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(dates, actuals, label="Actual", marker="o")
    plt.plot(dates, preds, label="Predicted", marker="x")
    plt.title(f"Bike Rentals: Actual vs Predicted ({start} to {end})")
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
