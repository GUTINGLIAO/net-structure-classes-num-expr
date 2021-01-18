import torch.nn as nn
from enum import Enum

from efficientnet_pytorch import EfficientNet


class EfficientNetType(Enum):
    B0 = EfficientNet.from_name('efficientnet-b0')
    B1 = EfficientNet.from_name('efficientnet-b1')
    B2 = EfficientNet.from_name('efficientnet-b2')
    B3 = EfficientNet.from_name('efficientnet-b3')
    B4 = EfficientNet.from_name('efficientnet-b4')
    B5 = EfficientNet.from_name('efficientnet-b5')
    B6 = EfficientNet.from_name('efficientnet-b6')
    B7 = EfficientNet.from_name('efficientnet-b7')


class VariantEfficientNet(nn.Module):
    """A variant of EfficientNet.

    This net is aimed to resolve the classification problem when we have different number of classes.
    So I add a full connection layer in the end to change output features. E.g, If I want to get a
    binary classfication net，the number of out features will be 2.
    """

    efficientnet: EfficientNet = ...
    fc: nn.Linear = ...

    def __init__(self, classes_num: int, efficient_net_type: EfficientNetType):
        super().__init__()
        self.efficientnet: EfficientNet = efficient_net_type.value
        self.fc = nn.Linear(1000, classes_num, bias=True)

    def forward(self, x):
        x = self.efficientnet.forward(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print(VariantEfficientNet(10, EfficientNetType.B1))
