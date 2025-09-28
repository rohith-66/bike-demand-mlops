import pandas as pd
import math

def _flag_us_holiday(d: pd.Series) -> pd.Series:
    # Simple fixed-date holiday set; good enough to boost signal
    mmdd = d.dt.strftime("%m-%d")
    fixed = {
        "01-01",  # New Year
        "07-04",  # Independence Day
        "12-24",  # Christmas Eve
        "12-25",  # Christmas
        "12-31",  # New Year's Eve
    }
    return mmdd.isin(fixed).astype(int)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Time-based
    df["year"] = df["dteday"].dt.year
    df["month"] = df["dteday"].dt.month
    df["dayofweek"] = df["dteday"].dt.dayofweek
    df["dayofyear"] = df["dteday"].dt.dayofyear

    # Cyclical (seasonality)
    df["sin_doy"] = df["dayofyear"].apply(lambda x: math.sin(2 * math.pi * x / 365.25))
    df["cos_doy"] = df["dayofyear"].apply(lambda x: math.cos(2 * math.pi * x / 365.25))

    # Lags / rollings
    df["lag_1"] = df["cnt"].shift(1)
    df["lag_7"] = df["cnt"].shift(7)
    df["rolling_7"] = df["cnt"].rolling(7).mean()
    df["rolling_30"] = df["cnt"].rolling(30).mean()

    # Booleans to ints
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_workingday"] = df["is_workingday"].astype(int)

    # === New: holiday/event & interactions ===
    df["is_us_holiday"] = _flag_us_holiday(df["dteday"])
    df["is_summer_peak"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_back_to_school"] = df["month"].isin([8, 9]).astype(int)

    df["is_wet"] = (df["precip_mm"] > 0).astype(int)
    df["temp_c_sq"] = df["temp_c"] ** 2
    df["windspeed_sq"] = df["windspeed"] ** 2

    df["temp_x_working"] = df["temp_c"] * df["is_workingday"]
    df["wet_x_working"] = df["is_wet"] * df["is_workingday"]

    return df

FEATURE_COLS = [
    "temp_c","humidity","windspeed","precip_mm",
    "is_holiday","is_workingday",
    "year","month","dayofweek","sin_doy","cos_doy",
    "lag_1","lag_7","rolling_7","rolling_30",
    "is_us_holiday","is_summer_peak","is_back_to_school",
    "is_wet","temp_c_sq","windspeed_sq","temp_x_working","wet_x_working"
]

TARGET_COL = "cnt"
DATE_COL = "dteday"
