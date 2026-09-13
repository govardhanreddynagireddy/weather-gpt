import xarray as xr
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

INPUT_FILE="ml/data/processed/era5_merged.nc"
OUTPUT_DIR="ml/data/processed"

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

TARGET="t2m"
SEQ_LEN=24

print("Loading ERA5 data...")

ds=xr.open_dataset(INPUT_FILE)

df=ds[FEATURES].to_dataframe().reset_index()

df=df.sort_values("valid_time").reset_index(drop=True)

print("Total rows:",len(df))
print("Time range:",df["valid_time"].min(),"to",df["valid_time"].max())

# Check hourly continuity
time_diff=df["valid_time"].diff().dropna()

print("Most common time difference:",time_diff.mode()[0])

if not (time_diff==pd.Timedelta(hours=1)).all():
    print("WARNING: Missing or irregular hourly timestamps!")

# --------------------------------------------------
# Chronological split
# --------------------------------------------------

n=len(df)

train_end=int(n*0.70)
val_end=int(n*0.85)

train_df=df.iloc[:train_end].copy()
val_df=df.iloc[train_end:val_end].copy()
test_df=df.iloc[val_end:].copy()

print("\nSplit:")
print("Train:",len(train_df))
print("Validation:",len(val_df))
print("Test:",len(test_df))

# --------------------------------------------------
# Scale using TRAINING data only
# --------------------------------------------------

feature_scaler=StandardScaler()
target_scaler=StandardScaler()

train_features=feature_scaler.fit_transform(train_df[FEATURES])
val_features=feature_scaler.transform(val_df[FEATURES])
test_features=feature_scaler.transform(test_df[FEATURES])

train_target=target_scaler.fit_transform(
    train_df[[TARGET]]
).flatten()

val_target=target_scaler.transform(
    val_df[[TARGET]]
).flatten()

test_target=target_scaler.transform(
    test_df[[TARGET]]
).flatten()

# --------------------------------------------------
# Create sequences
# --------------------------------------------------

def create_sequences(features,target,seq_len):
    X=[]
    y=[]

    for i in range(seq_len,len(features)):
        X.append(features[i-seq_len:i])
        y.append(target[i])

    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)


X_train,y_train=create_sequences(
    train_features,
    train_target,
    SEQ_LEN
)

X_val,y_val=create_sequences(
    val_features,
    val_target,
    SEQ_LEN
)

X_test,y_test=create_sequences(
    test_features,
    test_target,
    SEQ_LEN
)

print("\nSequence shapes:")
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_val:",X_val.shape)
print("y_val:",y_val.shape)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)

# --------------------------------------------------
# Save datasets
# --------------------------------------------------

np.savez_compressed(
    f"{OUTPUT_DIR}/train.npz",
    X=X_train,
    y=y_train
)

np.savez_compressed(
    f"{OUTPUT_DIR}/val.npz",
    X=X_val,
    y=y_val
)

np.savez_compressed(
    f"{OUTPUT_DIR}/test.npz",
    X=X_test,
    y=y_test
)

joblib.dump(
    feature_scaler,
    f"{OUTPUT_DIR}/feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    f"{OUTPUT_DIR}/target_scaler.pkl"
)

print("\nDataset preparation complete!")