import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# setup paths
project_root = Path(__file__).parent
os.makedirs(project_root / "models", exist_ok=True)

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

# load PySpark-engineered features
df = pd.read_csv(project_root / "data" / "tokyo_features.csv")

features = [
    "temperature_2m_lag_1h",
    "temperature_2m_lag_24h",
    "precipitation_lag_1h",
    "precipitation_lag_24h",

    "temperature_2m_24h_mean",
    "temperature_2m_24h_std",

    "humidity_24h_mean",

    "pressure_msl_24h_mean",
    "pressure_msl_24h_std",

    "precipitation_24h_sum",

    "wind_speed",
    "wind_speed_24h_mean",

    "precipitation_log",

    "hour_sin",
    "hour_cos",

    "month_sin",
    "month_cos",

    "is_weekend"
]

target = "temperature_2m"

# defensive feature check
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

X = df[features].values
y = df[target].values

# create multi-step sequences: predict next 168 hours (7 days)
horizon = 168

# snapshot-based forecasting with engineered temporal features
X_seq = X[:-horizon]
y_seq = np.array([
    y[i + 1:i + 1 + horizon]
    for i in range(len(X) - horizon)
])

# chronological train/test split
split_idx = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train.reshape(-1, 1)
).reshape(y_train.shape)

y_test = scaler_y.transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

# convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

dataset = TensorDataset(X_train_t, y_train_t)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

# probabilistic multi-horizon model
class WeatherForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, horizon=168):
        super().__init__()

        self.horizon = horizon

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, horizon * 2)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.horizon, 2)

# training configuration
n_ensembles = 3
models = []

print("Training Deep Ensemble Multi-Horizon Forecaster...")

for seed in range(n_ensembles):

    torch.manual_seed(seed)

    model = WeatherForecaster(
        input_dim=len(features),
        horizon=horizon
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.GaussianNLLLoss(reduction="mean")

    model.train()

    for epoch in range(25):

        epoch_loss = 0

        for xb, yb in loader:

            optimizer.zero_grad()

            out = model(xb)

            mean = out[:, :, 0]
            log_var = out[:, :, 1]

            var = torch.exp(log_var).clamp(
                min=1e-6,
                max=1e6
            )

            loss = criterion(mean, yb, var)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

    models.append(model)

    print(f"  Trained Ensemble Model {seed+1}/{n_ensembles}")

# evaluate on held-out test set
print("Evaluating On Held-Out Test Set...")

with torch.no_grad():

    means_test = []
    log_vars_test = []

    for m in models:

        m.eval()

        out = m(X_test_t)

        means_test.append(
            out[:, :, 0].cpu().numpy()
        )

        log_vars_test.append(
            out[:, :, 1].cpu().numpy()
        )

    y_pred_mean_test = np.mean(means_test, axis=0)

    y_pred_var_test = (
        np.mean(np.exp(log_vars_test), axis=0)
        + np.var(means_test, axis=0)
    )

    y_pred_std_test = np.sqrt(y_pred_var_test)

# inverse transform predictions
y_test_inv = scaler_y.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

y_pred_mean_inv_test = scaler_y.inverse_transform(
    y_pred_mean_test.reshape(-1, 1)
).reshape(y_pred_mean_test.shape)

y_pred_std_inv_test = (
    y_pred_std_test * scaler_y.scale_[0]
)

# compute metrics on first 24 forecast hours
hourly_pred_test = y_pred_mean_inv_test[:, :24]
hourly_std_test = y_pred_std_inv_test[:, :24]
hourly_true_test = y_test_inv[:, :24]

rmse_24h = np.sqrt(
    np.mean((hourly_true_test - hourly_pred_test) ** 2)
)

mae_24h = np.mean(
    np.abs(hourly_true_test - hourly_pred_test)
)

z = 1.96

lower_24h = hourly_pred_test - z * hourly_std_test
upper_24h = hourly_pred_test + z * hourly_std_test

coverage_24h = np.mean(
    (hourly_true_test >= lower_24h)
    & (hourly_true_test <= upper_24h)
)

# live-style forecast from most recent timestamp
print("\nGenerating Forecast From Most Recent Data Point...")

current_features = (
    df[features]
    .iloc[-1]
    .values
    .reshape(1, -1)
)

current_features = scaler_X.transform(current_features)

current_t = torch.tensor(
    current_features,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    preds_live = []

    for m in models:

        m.eval()

        out = m(current_t)

        mean = out[0, :, 0].cpu().numpy()
        log_var = out[0, :, 1].cpu().numpy()

        preds_live.append(
            (mean, np.exp(log_var))
        )

    mean_live = np.mean(
        [p[0] for p in preds_live],
        axis=0
    )

    var_live = (
        np.mean([p[1] for p in preds_live], axis=0)
        + np.var([p[0] for p in preds_live], axis=0)
    )

    std_live = np.sqrt(var_live)

    # inverse transform
    mean_inv_live = scaler_y.inverse_transform(
        mean_live.reshape(-1, 1)
    ).flatten()

    std_inv_live = (
        std_live * scaler_y.scale_[0]
    )

# ensure timestamp is datetime
df = df.sort_values("timestamp").reset_index(drop=True)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# create timestamps for next 168 hours
last_ts = df.iloc[-1]["timestamp"]

forecast_start = last_ts + pd.Timedelta(hours=1)

forecast_hours = pd.date_range(
    start=forecast_start,
    periods=168,
    freq="h"
)

# forecast dataframe
forecast_df = pd.DataFrame({
    "timestamp": forecast_hours,
    "pred_mean": mean_inv_live,
    "pred_std": std_inv_live,
    "lower_95": mean_inv_live - 1.96 * std_inv_live,
    "upper_95": mean_inv_live + 1.96 * std_inv_live
})

# save forecast
output_csv = (
    project_root
    / "data"
    / "latest_forecast.csv"
)

forecast_df.to_csv(output_csv, index=False)

print(f"\nForecast Results Saved To {output_csv}")

print(
    f"Forecast Period: "
    f"{forecast_hours[0]} "
    f"to "
    f"{forecast_hours[-1]}"
)

# weather condition classifier
def classify_weather(temp_std, precip_sum):

    if precip_sum > 5.0:
        return "Rainy"

    elif temp_std > 2.5:
        return "Variable"

    elif precip_sum > 1.0 or temp_std > 1.5:
        return "Partly Cloudy"

    else:
        return "Sunny"

# compute daily aggregates
mean_7d = mean_inv_live[:168].reshape(7, 24)
std_7d = std_inv_live[:168].reshape(7, 24)

daily_mean_live = mean_7d.mean(axis=1)
daily_max_live = mean_7d.max(axis=1)
daily_min_live = mean_7d.min(axis=1)

daily_std_live = std_7d.mean(axis=1)

# precipitation persistence proxy
precip_proxy = df.iloc[-1]["precipitation_24h_sum"]

daily_conditions = [
    classify_weather(
        std_7d[d].mean(),
        precip_proxy
    )
    for d in range(7)
]

# print results
print("\n" + "=" * 60)

print("FORECASTING RESULTS")

print("=" * 60)

print("Test Set Evaluation (Held-Out Chronological Data)")

print(
    f"   RMSE (Next 24h): {rmse_24h:.2f}C | "
    f"MAE: {mae_24h:.2f}C"
)

print(
    f"   95% Interval Coverage: "
    f"{coverage_24h:.1%}"
)

print("\nNext 24 Hours (Hourly Forecast)")

print(
    f"{'Hour Ahead':<12} "
    f"{'Pred Temp (C)':<15} "
    f"{'Uncertainty (sigma)':<15}"
)

print("-" * 45)

for h in range(24):

    print(
        f"{h+1:<12} "
        f"{mean_inv_live[h]:<15.1f} "
        f"+/-{std_inv_live[h]:<4.1f}C"
    )

print("\nNext 7 Days (Daily Overview)")

print(
    f"{'Day':<6} "
    f"{'Condition':<15} "
    f"{'Pred Mean':<10} "
    f"{'Pred Max':<10} "
    f"{'Pred Min':<10} "
    f"{'Uncertainty (sigma)':<15}"
)

print("-" * 75)

for d in range(7):

    print(
        f"Day {d+1:<3} "
        f"{daily_conditions[d]:<15} "
        f"{daily_mean_live[d]:<10.1f} "
        f"{daily_max_live[d]:<10.1f} "
        f"{daily_min_live[d]:<10.1f} "
        f"+/-{daily_std_live[d]:<4.1f}C"
    )

# save artifacts
for i, m in enumerate(models):

    torch.save(
        m.state_dict(),
        project_root
        / "models"
        / f"ensemble_model_{i}.pth"
    )

np.save(
    project_root / "models" / "scaler_X.npy",
    scaler_X
)

np.save(
    project_root / "models" / "scaler_y.npy",
    scaler_y
)

print("\nModels And Scalers Saved To models/")