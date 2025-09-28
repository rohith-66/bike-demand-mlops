import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
from src.features import add_time_features, FEATURE_COLS, TARGET_COL

MODEL_PATH = Path("models/bike_xgb_model.json")
DATA_PATH = Path("data/raw/day.csv")

def load_model():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster

def predict_for_date(date_str: str, model: xgb.Booster, df_raw: pd.DataFrame, override: dict | None = None) -> float:
    """
    Predict cnt for `date_str` using REAL lag/rolling features computed from history.
    If `override` is given, it can override weather/flags for the target date.
    """
    # Keep history up to the target date (inclusive)
    df_hist = df_raw.copy()
    df_hist["dteday"] = pd.to_datetime(df_hist["dteday"])
    target_date = pd.to_datetime(date_str)

    if override:
        # Apply overrides only for the target date row if present, else create it
        mask = df_hist["dteday"] == target_date
        if mask.any():
            for k, v in override.items():
                if k != "dteday":
                    df_hist.loc[mask, k] = v
        else:
            row = {"dteday": target_date, **override}
            # Ensure all required base columns exist
            for col in ["temp_c","humidity","windspeed","precip_mm","is_holiday","is_workingday","cnt"]:
                row.setdefault(col, 0)
            df_hist = pd.concat([df_hist, pd.DataFrame([row])], ignore_index=True)

    # Sort and cut history up to target
    df_hist = df_hist.sort_values("dteday")
    df_hist = df_hist[df_hist["dteday"] <= target_date].reset_index(drop=True)

    # Build features on the WHOLE history so lags/rollings are correct
    feats = add_time_features(df_hist)

    # Select the last row (the target date)
    row = feats.iloc[[-1]].copy()

    # Ensure column order matches model
    expected = model.feature_names
    for col in expected:
        if col not in row:
            row[col] = 0.0
    row = row[expected]

    dmat = xgb.DMatrix(row.values, feature_names=expected)
    yhat = float(model.predict(dmat)[0])
    return max(0.0, yhat)

if __name__ == "__main__":
    # Usage: python -m src.predict_from_history 2012-07-04
    date_arg = sys.argv[1] if len(sys.argv) > 1 else "2012-07-04"
    model = load_model()
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])

    yhat = predict_for_date(date_arg, model, df)
    print(f"Predicted rentals for {date_arg}: {yhat:.0f}")
