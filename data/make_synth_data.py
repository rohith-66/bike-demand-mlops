import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Folder to save the raw dataset
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def synth_day_data(start="2011-01-01", days=730, seed=42):
    rng = np.random.default_rng(seed)
    start_dt = datetime.fromisoformat(start)
    rows = []

    for i in range(days):
        dt = start_dt + timedelta(days=i)

        # Simulate weather & calendar
        day_of_year = dt.timetuple().tm_yday
        temp_c = 18 + 10*np.sin(2*np.pi*day_of_year/365.25) + rng.normal(0, 3)
        humidity = np.clip(rng.normal(0.55, 0.1), 0, 1)
        windspeed = np.clip(rng.normal(10, 5), 0, 40)
        precip = np.clip(rng.gamma(2.0, 1.0) - 1.5, 0, None)

        is_weekend = dt.weekday() >= 5
        is_holiday = (dt.month == 7 and dt.day == 4) or (dt.month == 12 and dt.day in (25, 31))
        is_workingday = (not is_weekend) and (not is_holiday)

        # Simulate demand count
        base = 250 + 3.5*temp_c - 60*humidity - 0.7*windspeed - 8*precip
        weekday_boost = 80 if is_workingday else -40
        seasonal = 120*np.sin(2*np.pi*(day_of_year-30)/365.25)
        noise = rng.normal(0, 30)

        count = max(0, int(base + weekday_boost + seasonal + noise))

        rows.append({
            "dteday": dt.date().isoformat(),
            "temp_c": round(float(temp_c), 2),
            "humidity": round(float(humidity), 3),
            "windspeed": round(float(windspeed), 2),
            "precip_mm": round(float(precip), 2),
            "is_holiday": int(is_holiday),
            "is_workingday": int(is_workingday),
            "cnt": count
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = synth_day_data()
    out = os.path.join(RAW_DIR, "day.csv")
    df.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(df)} rows")
