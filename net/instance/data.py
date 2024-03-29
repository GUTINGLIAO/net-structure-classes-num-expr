from enum import Enum
from typing import Any
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, VisionDataset

from net.instance.common_configuration import transform, DATASET_ROOT


class DatasetType(Enum):
    CIFAR10 = 1
    IMAGENET = 2
    CIFAR100 = 3

    # FIXME 是否有一种更好的获取枚举值方案
    @staticmethod
    def convert(num):
        for _type in DatasetType:
            if num == _type.value:
                return _type

        raise ValueError("%r is not a valid DatasetType" % num)


class DataLoaderFactory:
    """Factory to provide a data loader that has been processed.

    The main function is to filter classes and pack different datasets.
    """

    @classmethod
    def build(cls, dataset_type: DatasetType, train: bool, class_num: int, batch_size: int) -> DataLoader:
        if dataset_type == DatasetType.CIFAR10:
            data_set = CIFAR10(root=DATASET_ROOT, train=train, download=False,
                               transform=transform)
            cls._filter_classes(class_num, data_set)
            return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

        if dataset_type == DatasetType.IMAGENET:
            split: str = 'train' if train else 'val'
            data_set = ImageNet(root=DATASET_ROOT,
                                split=split,
                                transform=transform)
            cls._filter_classes(class_num, data_set)
            return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

        if dataset_type == DatasetType.CIFAR100:
            data_set = CIFAR100(root=DATASET_ROOT, train=train, download=False, transform=transform)
            cls._filter_classes(class_num, data_set)
            return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    @classmethod
    def _filter_classes(cls, classes_num, data_set):
        imgs = cls._get_imgs(data_set)
        # In all subclass of VisionDataset, targets have no other names, so there is no need to create a reference
        imgs_new: Any = []
        labels_new = []
        for i in range(len(data_set)):
            if data_set.targets[i] in tuple(range(classes_num)):
                imgs_new.append(imgs[i])
                labels_new.append(data_set.targets[i])

        classes = data_set.classes[0:classes_num]

        cls._replace(data_set, imgs_new, labels_new, classes)

    @classmethod
    def _replace(cls, data_set, imgs_new, labels_new, classes: list):
        if isinstance(data_set, CIFAR10):
            data_set.data, data_set.targets, data_set.classes = imgs_new, labels_new, classes
        if isinstance(data_set, ImageNet):
            data_set.imgs, data_set.targets, data_set.classes = imgs_new, labels_new, classes

    @classmethod
    def _get_imgs(cls, data_set: VisionDataset):
        """Get different imgs from different dataset.

        This method is aimed to deal with the different expression of imgs(image path and class index)
        in different subclass of VisionDataset. In CIFAR10, the imgs is called data. In ImageNet, it is
        called imgs.

        :param data_set: Different dataset.
        :return: List of (image path, class_index) tuples.
        """
        if isinstance(data_set, CIFAR10):
            return data_set.data
        if isinstance(data_set, ImageNet):
            return data_set.imgs
