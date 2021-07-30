import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,dropout):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv3d(6, 64, kernel_size=3, padding=1,stride=1),  # [64, 21,21,21]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),      # [64, 10,10,10]

            nn.Conv3d(64, 128, kernel_size=3, padding=1,stride=1), # [128, 10, 10,10]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),      # [128, 5, 5,5]

            nn.Conv3d(128, 256, kernel_size=3, padding=1,stride=1), # [256, 5, 5, 5]
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),      # [256, 2, 2, 2]
        )
        self.flatten=nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,1)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.flatten(out)
        return self.fc(out)