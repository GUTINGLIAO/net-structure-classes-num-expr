import torch.nn as nn

from enum import Enum

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet


class ResnetType(Enum):
    RESNET18 = resnet18()
    RESNET34 = resnet34()
    RESNET50 = resnet50()
    RESNET101 = resnet101()
    RESNET152 = resnet152()


class VariantResnet(nn.Module):

    resnet: ResNet
    fc: nn.Linear

    def name(self):
        return ""

    def __init__(self, classes_num: int, resnet_type: ResnetType):
        super().__init__()
        self.resnet: ResNet = resnet_type.value
        self.fc = nn.Linear(1000, classes_num, bias=True)

    def forward(self, x):
        x = self.resnet.forward(x)
        x = self.fc(x)
        return x
