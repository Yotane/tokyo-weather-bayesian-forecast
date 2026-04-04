import requests, pandas as pd, os, time
from datetime import datetime

os.makedirs("./data", exist_ok=True)

LAT, LON = 35.6762, 139.6503  # Tokyo
START_YEAR, END_YEAR = 1970, 2024
VARIABLES = "temperature_2m,precipitation"

all_dfs = []
for year in range(START_YEAR, END_YEAR + 1):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": VARIABLES, "timezone": "Asia/Tokyo"
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        data = res.json().get("hourly", {})
        if data:
            df = pd.DataFrame(data)
            df["year"] = year
            all_dfs.append(df)
            print(f"Fetched {year}: {len(df):,} rows")
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"Failed {year}: {e}")
        continue

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[["time", "temperature_2m", "precipitation", "year"]]
    final_df.to_csv("./data/tokyo_weather_1970_2024.csv", index=False)
    print(f"\nSaved {len(final_df):,} rows to data/tokyo_weather_1970_2024.csv")
else:
    print("No data fetched")