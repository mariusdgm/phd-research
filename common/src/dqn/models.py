import torch.autograd as autograd
import torch.nn as nn

from collections import OrderedDict

class Conv_QNET(nn.Module):
    def __init__(
        self,
        in_features,
        in_channels,
        num_actions,
        conv_hidden_out_size=16,
        lin_hidden_out_size=64,
    ):
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv_hidden_out_size = conv_hidden_out_size
        self.lin_hidden_out_size = lin_hidden_out_size

        # Convolutional layers with names
        self.features = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(
                    self.in_channels, self.conv_hidden_out_size, kernel_size=3, stride=1
                )),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(
                    self.conv_hidden_out_size,
                    self.conv_hidden_out_size,
                    kernel_size=3,
                    stride=1,
                )),
                ("relu2", nn.ReLU()),
            ])
        )

        # Linear layers with names
        self.fc = nn.Sequential(
            OrderedDict([
                ("lin1", nn.Linear(self.size_linear_unit(), self.lin_hidden_out_size)),
                ("relu3", nn.ReLU()),
                ("lin2", nn.Linear(self.lin_hidden_out_size, self.num_actions)),
            ])
        )
        

    def size_linear_unit(self):
        return (
            self.features(autograd.torch.zeros(*self.in_features)).view(1, -1).size(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv_QNET_one(nn.Module):
    def __init__(self, in_features, in_channels, num_actions, conv_hidden_out_size=16):
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv_hidden_out_size = conv_hidden_out_size

        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.conv_hidden_out_size, kernel_size=3, stride=1
            ),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.size_linear_unit(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def size_linear_unit(self):
        return (
            self.features(autograd.torch.zeros(*self.in_features)).view(1, -1).size(1)
        )

    def forward(self, x):
        # x = x.float()
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
