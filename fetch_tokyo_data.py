import requests, pandas as pd, os, time, numpy as np
from datetime import datetime

os.makedirs("./data", exist_ok=True)

LAT, LON = 35.6762, 139.6503  # Tokyo
START_YEAR, END_YEAR = 1970, 2024

# Open-Meteo API parameters (note: names differ slightly from your RAW_FEATURES)
VARIABLES = [
    "temperature_2m",           # to temperature_2m
    "precipitation",            # to precipitation
    "relative_humidity_2m",     # to humidity (rename later)
    "pressure_msl",             # to pressure_msl
    "wind_speed_10m",           # to convert to wind_u, wind_v
    "wind_direction_10m"        # to convert to wind_u, wind_v
]

def wind_to_uv(wind_speed_kmh, wind_dir_deg):
    # convert km/h to m/s
    speed_ms = np.array(wind_speed_kmh) / 3.6
    dir_rad = np.deg2rad(wind_dir_deg)
    # meteorological convention: wind FROM direction
    u = -speed_ms * np.sin(dir_rad)  # eastward component
    v = -speed_ms * np.cos(dir_rad)  # northward component
    return u, v

all_dfs = []
for year in range(START_YEAR, END_YEAR + 1):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(VARIABLES),
        "timezone": "Asia/Tokyo"
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json().get("hourly", {})
        if data and "time" in data:
            df = pd.DataFrame(data)
            # convert wind to u/v components
            if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
                df["wind_u"], df["wind_v"] = wind_to_uv(
                    df["wind_speed_10m"].fillna(0),
                    df["wind_direction_10m"].fillna(0)
                )

            df = df.rename(columns={
                "relative_humidity_2m": "humidity"
            })
            df["year"] = year
            all_dfs.append(df)
            print(f"Fetched {year}: {len(df):,} rows")
        time.sleep(0.5)
    except Exception as e:
        print(f"Failed {year}: {e}")
        continue

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    # select and order columns to match your RAW_FEATURES + target
    cols_to_keep = ["time", "temperature_2m", "precipitation", "humidity", 
                   "pressure_msl", "wind_u", "wind_v", "year"]
    # only keep columns that exist (defensive)
    cols_to_keep = [c for c in cols_to_keep if c in final_df.columns]
    final_df = final_df[cols_to_keep]
    final_df.to_csv("./data/tokyo_weather_1970_2024.csv", index=False)
    print(f"\nSaved {len(final_df):,} rows to data/tokyo_weather_1970_2024.csv")
    print(f"Columns: {list(final_df.columns)}")
else:
    print("No data fetched")