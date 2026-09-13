import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",DEVICE)


# Load test data
data=np.load("ml/data/processed/test.npz")

X_test=torch.tensor(data["X"],dtype=torch.float32)
y_test=torch.tensor(data["y"],dtype=torch.float32)

print("X_test:",X_test.shape)
print("y_test:",y_test.shape)


# GRU model
class WeatherGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru=nn.GRU(
            input_size=8,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc=nn.Linear(128,1)

    def forward(self,x):

        output,hidden=self.gru(x)

        x=output[:,-1,:]

        x=self.fc(x)

        return x.squeeze(1)


# Load trained model
model=WeatherGRU().to(DEVICE)

model.load_state_dict(
    torch.load(
        "ml/data/processed/best_weather_gru.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


# Prediction
with torch.no_grad():

    X_test=X_test.to(DEVICE)

    predictions=model(X_test)

    predictions=predictions.cpu().numpy()

y_test=y_test.numpy()


# Metrics
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)


print()
print("==============================")
print("GRU MODEL EVALUATION")
print("==============================")

print("MSE :",mse)
print("RMSE:",rmse)
print("MAE :",mae)
print("R2  :",r2)

print("==============================")