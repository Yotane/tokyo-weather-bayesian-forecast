import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# setup paths
project_root = Path(__file__).parent
figures_dir = project_root / "figures"
figures_dir.mkdir(exist_ok=True)

# load forecast results
forecast_csv = project_root / "data" / "latest_forecast.csv"
if not forecast_csv.exists():
    raise FileNotFoundError(f"Please run 03_bayesian_forecasting.py first to generate {forecast_csv}")

df = pd.read_csv(forecast_csv)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# PLOT 1: NEXT 24 HOURS
plt.figure(figsize=(12, 6))
data_24h = df.iloc[:24]

plt.plot(data_24h['timestamp'], data_24h['pred_mean'], 
         label='Predicted Temp', color='#1f77b4', linewidth=2, marker='o', markersize=4)
plt.fill_between(data_24h['timestamp'], 
                 data_24h['lower_95'], 
                 data_24h['upper_95'], 
                 color='#1f77b4', alpha=0.2, label='95% Confidence Interval')

plt.title('Tokyo Temperature Forecast: Next 24 Hours', fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(figures_dir / "forecast_24h.png", dpi=150, bbox_inches='tight')
print("Saved 24-hour forecast plot to figures/forecast_24h.png")

# PLOT 2: NEXT 7 DAYS
plt.figure(figsize=(12, 6))
data_7d = df.iloc[:168].copy()
data_7d['day'] = data_7d['timestamp'].dt.floor('D')

# aggregate daily stats
daily_stats = data_7d.groupby('day').agg(
    mean_temp=('pred_mean', 'mean'),
    max_temp=('pred_mean', 'max'),
    min_temp=('pred_mean', 'min'),
    std_temp=('pred_std', 'mean') # average hourly uncertainty for the day
).reset_index()

x_pos = np.arange(len(daily_stats))
bar_width = 0.6

plt.bar(x_pos, daily_stats['mean_temp'], yerr=1.96*daily_stats['std_temp'], 
        capsize=5, color='#2ca02c', alpha=0.7, label='Daily Mean ± 95% CI')
plt.scatter(x_pos, daily_stats['max_temp'], color='red', marker='^', s=100, zorder=5, label='Daily Max')
plt.scatter(x_pos, daily_stats['min_temp'], color='blue', marker='v', s=100, zorder=5, label='Daily Min')

plt.title('Tokyo Temperature Forecast: Next 7 Days', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(x_pos, [d.strftime('%m-%d') for d in daily_stats['day']], rotation=45)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(figures_dir / "forecast_7d.png", dpi=150, bbox_inches='tight')
print("Saved 7-day forecast plot to figures/forecast_7d.png")

plt.show()