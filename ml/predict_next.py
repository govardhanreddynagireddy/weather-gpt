import torch
import numpy as np
import xarray as xr
import joblib

from model import WeatherGRU


# ==============================
# Configuration
# ==============================

MODEL_PATH="ml/data/processed/best_weather_gru.pth"
DATA_PATH="ml/data/processed/era5_merged.nc"
FEATURE_SCALER_PATH="ml/data/processed/feature_scaler.pkl"
TARGET_SCALER_PATH="ml/data/processed/target_scaler.pkl"

FEATURES=[
    "d2m",
    "t2m",
    "sp",
    "tp",
    "ssrd",
    "skt",
    "u10",
    "v10"
]

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:",DEVICE)

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))


# ==============================
# Load model
# ==============================

model=WeatherGRU(
    input_size=8,
    hidden_size=128,
    num_layers=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# ==============================
# Load scalers
# ==============================

feature_scaler=joblib.load(
    FEATURE_SCALER_PATH
)

target_scaler=joblib.load(
    TARGET_SCALER_PATH
)

print("Scalers loaded successfully!")


# ==============================
# Load ERA5 data
# ==============================

ds=xr.open_dataset(DATA_PATH)

df=ds[FEATURES].to_dataframe()

print("Total records:",len(df))

# Make sure data is sorted
df=df.sort_index()


# ==============================
# Take latest 24 hours
# ==============================

latest=df.iloc[-24:][FEATURES].values

print("Latest sequence shape:",latest.shape)

if len(latest)!=24:
    raise ValueError("Not enough data for 24-hour sequence")


# ==============================
# Scale features
# ==============================

latest_scaled=feature_scaler.transform(
    latest
)


# ==============================
# Prepare tensor
# ==============================

x=torch.tensor(
    latest_scaled,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)


# ==============================
# GRU Prediction
# ==============================

with torch.no_grad():

    prediction=model(x)


# ==============================
# Convert scaled prediction
# ==============================

prediction_scaled=prediction.cpu().numpy().reshape(-1,1)

prediction_kelvin=target_scaler.inverse_transform(
    prediction_scaled
)[0][0]


# Kelvin → Celsius
prediction_celsius=prediction_kelvin-273.15


# ==============================
# Latest timestamp
# ==============================

latest_time=df.index[-1]


# ==============================
# Display result
# ==============================

print()
print("==============================")
print("NEXT HOUR WEATHER PREDICTION")
print("==============================")

print("Latest ERA5 time:",latest_time)

print(
    f"Predicted Temperature: "
    f"{prediction_celsius:.2f} °C"
)

print("==============================")