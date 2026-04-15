import torch
import torch.nn as nn

class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
