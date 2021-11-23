import torch.nn as nn
import torch


class DPEN(nn.Module):

    def __init__(self):
        super(DPEN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2,3])
        x = self.fc(x)
        return x[:, 0], x[:, 1]

