import torch.nn as nn


class QNET(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(QNET, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
        )

    def forward(self, x):
        return self.network(x)
