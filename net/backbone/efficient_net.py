import torch.nn as nn
from enum import Enum

from efficientnet_pytorch import EfficientNet


class EfficientNetType(Enum):
    EFFICIENT_NET_B0 = 1
    EFFICIENT_NET_B1 = 2
    EFFICIENT_NET_B2 = 3
    EFFICIENT_NET_B3 = 4
    EFFICIENT_NET_B4 = 5
    EFFICIENT_NET_B5 = 6
    EFFICIENT_NET_B6 = 7
    EFFICIENT_NET_B7 = 8

    # FIXME 是否有一种更好的获取枚举值方案
    @staticmethod
    def convert(num):
        for _type in EfficientNetType:
            if num == _type.value:
                return _type

        raise ValueError("%r is not a valid EfficientNetType" % num)


_efficient_net_dict = {
    EfficientNetType.EFFICIENT_NET_B0: EfficientNet.from_name('efficientnet-b0'),
    EfficientNetType.EFFICIENT_NET_B1: EfficientNet.from_name('efficientnet-b1'),
    EfficientNetType.EFFICIENT_NET_B2: EfficientNet.from_name('efficientnet-b2'),
    EfficientNetType.EFFICIENT_NET_B3: EfficientNet.from_name('efficientnet-b3'),
    EfficientNetType.EFFICIENT_NET_B4: EfficientNet.from_name('efficientnet-b4'),
    EfficientNetType.EFFICIENT_NET_B5: EfficientNet.from_name('efficientnet-b5'),
    EfficientNetType.EFFICIENT_NET_B6: EfficientNet.from_name('efficientnet-b6'),
    EfficientNetType.EFFICIENT_NET_B7: EfficientNet.from_name('efficientnet-b7')
}


class VariantEfficientNet(nn.Module):
    """A variant of EfficientNet.

    This net is aimed to resolve the classification problem when we have different number of classes.
    So I add a full connection layer in the end to change output features. E.g, If I want to get a
    binary classfication net，the number of out features will be 2.
    """

    efficientnet: EfficientNet
    fc: nn.Linear

    def __init__(self, classes_num: int, efficient_net_type: EfficientNetType):
        super().__init__()
        self.efficientnet: EfficientNet = _efficient_net_dict[efficient_net_type]
        self.fc = nn.Linear(1000, classes_num, bias=True)

    def forward(self, x):
        x = self.efficientnet.forward(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print(VariantEfficientNet(10, EfficientNetType.B1))
