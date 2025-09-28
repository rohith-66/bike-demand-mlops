import pandas as pd
import xgboost as xgb
from pathlib import Path
from src.features import add_time_features, FEATURE_COLS

MODEL_PATH = Path("models/bike_xgb_model.json")

def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

def predict_one(payload: dict, model):
    # Convert dict → DataFrame
    df = pd.DataFrame([payload])
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Add engineered features
    df = add_time_features(df)

    # Fill lag/rolling NaNs (since we don’t have history in manual prediction)
    for col in ["lag_1","lag_7","rolling_7","rolling_30"]:
        if col in df:
            df[col] = df[col].fillna(0.0)

    # Align with model’s expected features
    expected = model.feature_names
    for col in expected:
        if col not in df:
            df[col] = 0.0
    df = df[expected]

    # Predict
    dmat = xgb.DMatrix(df.values, feature_names=expected)
    yhat = float(model.predict(dmat)[0])
    return max(0.0, yhat)

if __name__ == "__main__":
    model = load_model()

    # Example: July 4th holiday, hot & sunny
    payload = {
    "dteday": "2012-07-04",
    "temp_c": 19.07,
    "humidity": 0.756,
    "windspeed": 14.7,
    "precip_mm": 1.77,
    "is_holiday": 1,
    "is_workingday": 0,
    "cnt": 0
}


    yhat = predict_one(payload, model)
    print(f"Predicted rentals for {payload['dteday']}: {yhat:.0f}")
