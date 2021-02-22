import torch.nn as nn

from enum import Enum

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet


# TODO 使用继承或者组合提取枚举的打印逻辑和转换逻辑
class ResnetType(Enum):
    RESNET18 = 1
    RESNET34 = 2
    RESNET50 = 3
    RESNET101 = 4
    RESNET152 = 5

    # FIXME 是否有一种更好的获取枚举值方案
    @staticmethod
    def convert(num):
        for _type in ResnetType:
            if num == _type.value:
                return _type

        raise ValueError("%r is not a valid ResnetType" % num)


_resnet_dict = {
    ResnetType.RESNET18: resnet18(),
    ResnetType.RESNET34: resnet34(),
    ResnetType.RESNET50: resnet50(),
    ResnetType.RESNET101: resnet101(),
    ResnetType.RESNET152: resnet152()
}


class VariantResnet(nn.Module):
    resnet: ResNet
    fc: nn.Linear

    def name(self):
        return ""

    def __init__(self, classes_num: int, resnet_type: ResnetType):
        super().__init__()
        self.resnet: ResNet = _resnet_dict[resnet_type]
        self.fc = nn.Linear(1000, classes_num, bias=True)

    def forward(self, x):
        x = self.resnet.forward(x)
        x = self.fc(x)
        return x
