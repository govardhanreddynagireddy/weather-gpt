import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Guard against native access violation in PyTorch CUDA stream capture on CPU/Windows
import torch
if hasattr(torch, "cuda"):
    torch.cuda.is_current_stream_capturing = lambda: False
    if hasattr(torch.cuda, "graphs"):
        torch.cuda.graphs.is_current_stream_capturing = lambda: False

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import xarray as xr


# =====================================================
# PATHS
# =====================================================

BASE_DIR=Path(__file__).resolve().parents[2]

MODEL_PATH=BASE_DIR/"ml/data/processed/best_weather_gru.pth"

FEATURE_SCALER_PATH=BASE_DIR/"ml/data/processed/feature_scaler.pkl"

TARGET_SCALER_PATH=BASE_DIR/"ml/data/processed/target_scaler.pkl"

TEST_DATA_PATH=BASE_DIR/"ml/data/processed/test.npz"

ERA5_DATA_PATH=BASE_DIR/"ml/data/processed/era5_merged.nc"


# =====================================================
# SETTINGS
# =====================================================

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

SEQ_LEN=24

HIDDEN_SIZE=128

NUM_LAYERS=2

DROPOUT=0.2

TARGET="t2m"

TARGET_UNIT="celsius"

PREDICTION_HORIZON="next_hour"


# =====================================================
# DEVICE
# =====================================================

def _get_device():
    return torch.device("cpu")

device=_get_device()


# =====================================================
# GRU MODEL
# =====================================================

class WeatherGRU(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ):

        super().__init__()

        self.gru=nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=DROPOUT
        )

        self.fc=nn.Linear(
            hidden_size,
            1
        )


    def forward(self,x):

        output,_=self.gru(x)

        last=output[:,-1,:]

        return self.fc(last).squeeze(1)


# =====================================================
# LAZY-LOADED GRU STATE
# =====================================================

model=None
feature_scaler=None
target_scaler=None


def _ensure_gru_loaded():

    global model,feature_scaler,target_scaler

    if (
        model is not None
        and feature_scaler is not None
        and target_scaler is not None
    ):
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GRU model not found: {MODEL_PATH}"
        )

    if not FEATURE_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Feature scaler not found: {FEATURE_SCALER_PATH}"
        )

    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Target scaler not found: {TARGET_SCALER_PATH}"
        )

    print("Loading WeatherGRU...")

    dev=_get_device()

    m=WeatherGRU().to(dev)

    m.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=dev
        )
    )

    m.eval()

    fs=joblib.load(
        FEATURE_SCALER_PATH
    )

    ts=joblib.load(
        TARGET_SCALER_PATH
    )

    model=m
    feature_scaler=fs
    target_scaler=ts

    print("WeatherGRU ready")


# =====================================================
# PREDICT TEMPERATURE
# =====================================================

def predict_temperature(weather_data):

    """
    Predict next-hour temperature.

    Input shape:
        24 x 8

    Feature order:
        d2m
        t2m
        sp
        tp
        ssrd
        skt
        u10
        v10

    ERA5 t2m target is stored in Kelvin.

    Returned prediction is Celsius.
    """

    _ensure_gru_loaded()

    # -------------------------------------------------
    # Convert input to NumPy
    # -------------------------------------------------

    data=np.asarray(
        weather_data,
        dtype=np.float32
    )


    # -------------------------------------------------
    # Validate shape
    # -------------------------------------------------

    expected_shape=(
        SEQ_LEN,
        len(FEATURES)
    )

    if data.shape!=expected_shape:

        raise ValueError(
            f"Expected shape {expected_shape} "
            f"but received {data.shape}"
        )


    # -------------------------------------------------
    # Create DataFrame
    #
    # This prevents:
    # "X does not have valid feature names"
    # -------------------------------------------------

    data_df=pd.DataFrame(
        data,
        columns=FEATURES
    )


    # -------------------------------------------------
    # Scale features
    # -------------------------------------------------

    data_scaled=feature_scaler.transform(
        data_df
    )


    # -------------------------------------------------
    # Convert to PyTorch
    # -------------------------------------------------

    X=torch.tensor(
        data_scaled,
        dtype=torch.float32
    )

    X=X.unsqueeze(0)

    X=X.to(device)


    # -------------------------------------------------
    # GRU prediction
    # -------------------------------------------------

    with torch.no_grad():

        prediction_scaled=model(X)


    # -------------------------------------------------
    # Move to CPU
    # -------------------------------------------------

    prediction_scaled=(
        prediction_scaled
        .cpu()
        .numpy()
        .reshape(-1,1)
    )


    # -------------------------------------------------
    # Inverse target scaling
    # -------------------------------------------------

    prediction=target_scaler.inverse_transform(
        prediction_scaled
    )


    # -------------------------------------------------
    # Kelvin → Celsius
    # -------------------------------------------------

    temperature_kelvin=float(
        prediction[0][0]
    )

    temperature_celsius=(
        temperature_kelvin-273.15
    )


    # -------------------------------------------------
    # Return Celsius
    # -------------------------------------------------

    return float(
        temperature_celsius
    )


# =====================================================
# PREDICT FROM TEST DATA
# =====================================================

def predict_test_sample(index=0):

    """
    Test the trained GRU using
    one sequence from test.npz.
    """

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test data not found: "
            f"{TEST_DATA_PATH}"
        )


    test=np.load(
        TEST_DATA_PATH
    )

    X=test["X"]


    # -------------------------------------------------
    # Validate index
    # -------------------------------------------------

    if index<0 or index>=len(X):

        raise IndexError(
            f"Sample index must be between "
            f"0 and {len(X)-1}"
        )


    # -------------------------------------------------
    # Get 24-hour sequence
    # -------------------------------------------------

    sequence=X[index]


    # -------------------------------------------------
    # Predict
    # -------------------------------------------------

    prediction=predict_temperature(
        sequence
    )


    # -------------------------------------------------
    # Return
    # -------------------------------------------------

    return {

        "prediction_celsius":round(
            prediction,
            2
        ),

        "sample_index":index,

        "sequence_length":SEQ_LEN,

        "feature_count":len(FEATURES),

        "source":"test.npz",

        "prediction_horizon":
            PREDICTION_HORIZON

    }


# =====================================================
# PREDICT FROM LATEST ERA5 DATA
# =====================================================

def predict_latest_temperature():

    """
    Predict the next-hour temperature using
    the latest 24 valid observations from
    era5_merged.nc.
    """

    # -------------------------------------------------
    # Check ERA5 file
    # -------------------------------------------------

    if not ERA5_DATA_PATH.exists():

        raise FileNotFoundError(
            f"ERA5 data not found: "
            f"{ERA5_DATA_PATH}"
        )


    print("Loading latest ERA5 data...")


    # -------------------------------------------------
    # Open dataset
    # -------------------------------------------------

    dataset=xr.open_dataset(
        ERA5_DATA_PATH
    )


    try:

        # ---------------------------------------------
        # Check features
        # ---------------------------------------------

        missing=[]

        for feature in FEATURES:

            if feature not in dataset:

                missing.append(
                    feature
                )


        if missing:

            raise ValueError(
                "Missing features in ERA5 dataset: "
                f"{missing}"
            )


        # ---------------------------------------------
        # Extract features
        # ---------------------------------------------

        feature_arrays=[]

        for feature in FEATURES:

            values=dataset[feature].values

            values=np.asarray(
                values,
                dtype=np.float32
            )

            values=values.reshape(-1)

            feature_arrays.append(
                values
            )


        # ---------------------------------------------
        # Combine into table
        # ---------------------------------------------

        data=np.column_stack(
            feature_arrays
        )


        # ---------------------------------------------
        # Remove NaN / infinite rows
        # ---------------------------------------------

        valid_rows=(
            np.isfinite(data).all(
                axis=1
            )
        )

        data=data[valid_rows]


        # ---------------------------------------------
        # Check enough data
        # ---------------------------------------------

        if len(data)<SEQ_LEN:

            raise ValueError(
                f"Need at least {SEQ_LEN} "
                f"valid observations in ERA5 data, "
                f"but only {len(data)} available."
            )


        # ---------------------------------------------
        # Latest 24 observations
        # ---------------------------------------------

        sequence=data[-SEQ_LEN:]


        # ---------------------------------------------
        # Predict
        # ---------------------------------------------

        prediction=predict_temperature(
            sequence
        )


        # ---------------------------------------------
        # Return result
        # ---------------------------------------------

        return {

            "prediction_celsius":round(
                prediction,
                2
            ),

            "sequence_length":SEQ_LEN,

            "feature_count":len(FEATURES),

            "source":"era5_merged.nc",

            "prediction_horizon":
                PREDICTION_HORIZON

        }


    finally:

        dataset.close()


# =====================================================
# MODEL INFORMATION
# =====================================================

def get_gru_info():

    return {

        "model":"WeatherGRU",

        "sequence_length":SEQ_LEN,

        "features":FEATURES,

        "feature_count":len(FEATURES),

        "hidden_size":HIDDEN_SIZE,

        "num_layers":NUM_LAYERS,

        "dropout":DROPOUT,

        "target":TARGET,

        "target_unit":TARGET_UNIT,

        "prediction_horizon":
            PREDICTION_HORIZON,

        "device":str(device)

    }


# =====================================================
# DIRECT TEST
# =====================================================

if __name__=="__main__":

    print()

    print("========================================")

    print("WeatherGPT+ GRU Test")

    print("========================================")

    print()


    # -------------------------------------------------
    # Model information
    # -------------------------------------------------

    print("Model information:")

    print(
        get_gru_info()
    )

    print()


    # -------------------------------------------------
    # Test sample
    # -------------------------------------------------

    print(
        "Testing test.npz sample 0..."
    )

    result=predict_test_sample(
        index=0
    )

    print()

    print(
        "Test sample prediction:"
    )

    print(
        result
    )

    print()


    # -------------------------------------------------
    # Latest ERA5 prediction
    # -------------------------------------------------

    print(
        "Testing latest ERA5 data..."
    )

    latest=predict_latest_temperature()

    print()

    print(
        "Latest ERA5 prediction:"
    )

    print(
        latest
    )

    print()

    print(
        "GRU test completed."
    )