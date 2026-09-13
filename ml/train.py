import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

# -----------------------------
# Device
# -----------------------------

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))

# -----------------------------
# Load data
# -----------------------------

train=np.load("ml/data/processed/train.npz")
val=np.load("ml/data/processed/val.npz")

X_train=torch.tensor(train["X"],dtype=torch.float32)
y_train=torch.tensor(train["y"],dtype=torch.float32)

X_val=torch.tensor(val["X"],dtype=torch.float32)
y_val=torch.tensor(val["y"],dtype=torch.float32)

print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_val:",X_val.shape)
print("y_val:",y_val.shape)

# -----------------------------
# DataLoader
# -----------------------------

train_dataset=TensorDataset(X_train,y_train)
val_dataset=TensorDataset(X_val,y_val)

train_loader=DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

val_loader=DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False
)

# -----------------------------
# GRU Model
# -----------------------------

class WeatherGRU(nn.Module):

    def __init__(self,input_size=8,hidden_size=128,num_layers=2):
        super().__init__()

        self.gru=nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc=nn.Linear(hidden_size,1)

    def forward(self,x):

        output,_=self.gru(x)

        # Last 24-hour hidden state
        last=output[:,-1,:]

        return self.fc(last).squeeze(1)


model=WeatherGRU().to(device)

print("\nModel:")
print(model)

# -----------------------------
# Loss and optimizer
# -----------------------------

criterion=nn.MSELoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# Training
# -----------------------------

epochs=20

best_val_loss=float("inf")

for epoch in range(epochs):

    model.train()

    train_loss=0

    for X,y in train_loader:

        X=X.to(device)
        y=y.to(device)

        optimizer.zero_grad()

        predictions=model(X)

        loss=criterion(predictions,y)

        loss.backward()

        optimizer.step()

        train_loss+=loss.item()*X.size(0)

    train_loss/=len(train_loader.dataset)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss=0

    with torch.no_grad():

        for X,y in val_loader:

            X=X.to(device)
            y=y.to(device)

            predictions=model(X)

            loss=criterion(predictions,y)

            val_loss+=loss.item()*X.size(0)

    val_loss/=len(val_loader.dataset)

    print(
        f"Epoch {epoch+1:02d}/{epochs} "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    # -------------------------
    # Save best model
    # -------------------------

    if val_loss<best_val_loss:

        best_val_loss=val_loss

        torch.save(
            model.state_dict(),
            "ml/data/processed/best_weather_gru.pth"
        )

        print("  Saved best model!")

print("\nTraining complete!")
print("Best validation loss:",best_val_loss)
print("Model saved:")
print("ml/data/processed/best_weather_gru.pth")