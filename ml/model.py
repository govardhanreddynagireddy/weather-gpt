import torch
import torch.nn as nn


class WeatherGRU(nn.Module):

    def __init__(self,input_size=8,hidden_size=128,num_layers=2):

        super().__init__()

        self.gru=nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc=nn.Linear(hidden_size,1)


    def forward(self,x):

        output,hidden=self.gru(x)

        x=output[:,-1,:]

        x=self.fc(x)

        return x.squeeze(1)