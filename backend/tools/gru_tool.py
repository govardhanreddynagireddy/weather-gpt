from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib


# =====================================================
# PATHS
# =====================================================

BASE_DIR=Path(__file__).resolve().parents[2]

MODEL_PATH=BASE_DIR/"ml/data/processed/best_weather_gru.pth"

FEATURE_SCALER_PATH=BASE_DIR/"ml/data/processed/feature_scaler.pkl"

TARGET_SCALER_PATH=BASE_DIR/"ml/data/processed/target_scaler.pkl"

TEST_DATA_PATH=BASE_DIR/"ml/data/processed/test.npz"


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


# =====================================================
# DEVICE
# =====================================================

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# GRU MODEL
# =====================================================

class WeatherGRU(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden_size=128,
        num_layers=2
    ):

        super().__init__()

        self.gru=nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
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
# CHECK REQUIRED FILES
# =====================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"GRU model not found: {MODEL_PATH}"
    )


if not FEATURE_SCALER_PATH.exists():

    raise FileNotFoundError(
        f"Feature scaler not found: "
        f"{FEATURE_SCALER_PATH}"
    )


if not TARGET_SCALER_PATH.exists():

    raise FileNotFoundError(
        f"Target scaler not found: "
        f"{TARGET_SCALER_PATH}"
    )


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading WeatherGRU...")

model=WeatherGRU().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# =====================================================
# LOAD SCALERS
# =====================================================

feature_scaler=joblib.load(
    FEATURE_SCALER_PATH
)

target_scaler=joblib.load(
    TARGET_SCALER_PATH
)


print("WeatherGRU ready")


# =====================================================
# PREDICT TEMPERATURE
# =====================================================

def predict_temperature(
    weather_data
):

    """
    Predict next-hour t2m.

    Input:

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
    """


    # -------------------------------------------------
    # Convert input to NumPy
    # -------------------------------------------------

    data=np.asarray(
        weather_data,
        dtype=np.float32
    )


    # -------------------------------------------------
    # Validate input
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
    # Scale features
    # -------------------------------------------------

    data_scaled=feature_scaler.transform(
        data
    )


    # -------------------------------------------------
    # Convert to PyTorch tensor
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
    # Move prediction to CPU
    # -------------------------------------------------

    prediction_scaled=(
        prediction_scaled
        .cpu()
        .numpy()
        .reshape(-1,1)
    )


    # -------------------------------------------------
    # Inverse scaling
    # -------------------------------------------------

    prediction=target_scaler.inverse_transform(
        prediction_scaled
    )


    # -------------------------------------------------
    # Return Celsius
    # -------------------------------------------------

    return float(
        prediction[0][0]
    )


# =====================================================
# PREDICT FROM TEST DATA
# =====================================================

def predict_test_sample(
    index=0
):

    """
    Test the trained GRU using
    a sequence from test.npz.
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


    return {

        "prediction_celsius":round(
            prediction,
            2
        ),

        "sample_index":index,

        "sequence_length":SEQ_LEN,

        "feature_count":len(FEATURES)

    }


# =====================================================
# MODEL INFORMATION
# =====================================================

def get_gru_info():

    return {

        "model":"WeatherGRU",

        "sequence_length":SEQ_LEN,

        "features":FEATURES,

        "feature_count":len(FEATURES),

        "hidden_size":128,

        "num_layers":2,

        "dropout":0.2,

        "target":"t2m",

        "prediction_horizon":"next_hour",

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

    print(
        "Model information:"
    )

    print(
        get_gru_info()
    )

    print()

    print(
        "Testing sample 0..."
    )

    result=predict_test_sample(
        index=0
    )

    print()

    print(
        "Prediction:"
    )

    print(
        result
    )

    print()

    print(
        "GRU test completed."
    )