from torchvision import transforms as transforms
from torch.nn import CrossEntropyLoss, Module
from pathlib import Path

__all__ = ('DATASET_ROOT', 'transform', 'loss_criterion')

# TODO Create soft link of image net
DATASET_ROOT = Path.joinpath(Path(__file__).parents[2], 'data')
IMAGE_ROOT = '/ai/imageNet数据集'

# Normalize images in dataset from [0,1] to [-1,-1].
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

loss_criterion: Module = CrossEntropyLoss()

if __name__ == '__main__':
    print(DATASET_ROOT)

