from enum import Enum
from typing import Any
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10

from net.instance.common_configuration import transform, DATASET_ROOT

class DatasetType(Enum):
    CIFAR10 = 0
    IMAGENET = 1


class DataLoaderFactory:
    """Factory to provide a data loader that has been processed.

    The
    """

    @classmethod
    def build(cls, dataset_type: DatasetType, train: bool, class_num: int) -> DataLoader:
        if dataset_type == DatasetType.CIFAR10:
            data_set = CIFAR10(root=DATASET_ROOT, train=train, download=False,
                               transform=transform)
            cls._filter_classes(class_num, data_set)
            return DataLoader(data_set, batch_size=4, shuffle=True, num_workers=2)

    @classmethod
    def _filter_classes(cls, class_num, data_set):
        data_new: Any = []
        targets_new = []
        for i in range(data_set.__len__()):
            if data_set.targets[i] in tuple(range(class_num)):
                data_new.append(data_set.data[i])
                targets_new.append(data_set.targets[i])
        data_set.data, data_set.targets = data_new, targets_new
