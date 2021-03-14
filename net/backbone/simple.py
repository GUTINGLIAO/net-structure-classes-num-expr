import torch.nn as nn
import torch.nn.functional as functional


class SimpleNet(nn.Module):
    """Structure of the neural networks."""

    def __init__(self, classes_num: int):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)

    def forward(self, x):
        """ Forword calculate through the network.

        Something you should know is that this method is implemented by Father Class, which is written by C++.

        :param x: input
        :return: final result
        """
        x = self.pool(functional.relu(self.conv1(x)))

        x = self.pool(functional.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = functional.relu(self.fc1(x))

        x = functional.relu(self.fc2(x))

        x = self.fc3(x)

        return x
