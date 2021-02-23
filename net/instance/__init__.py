from enum import Enum

from net.cnn import Cnn
from net.instance.data import DataLoaderFactory, DatasetType
from net.instance.common_configuration import loss_criterion
from net.structure.efficient_net import VariantEfficientNet, EfficientNetType
from net.structure.resnet import VariantResnet, ResnetType

__all__ = ['NetBuilder', 'NetType', 'simple_cnn_cifar10_instance_2_classes',
           'simple_cnn_cifar10_instance_3_classes', 'simple_cnn_cifar10_instance_4_classes',
           'simple_cnn_cifar10_instance_5_classes', 'simple_cnn_cifar10_instance_10_classes',
           'resnet_18_cifar10_instance_2_classes', 'instance_dict']

from net.structure.simple import SimpleNet


class NetType(Enum):
    SIMPLE_NET = 1
    RESNET = 2
    EFFICIENT_NET = 3


# TODO 继续提升可扩展性，使新增一种网络结构或者数据集更加简单
class NetBuilder:

    @classmethod
    def efficient_net_instance(cls, data_set_type: DatasetType, efficient_net_type: EfficientNetType,
                               classes_num: int, epoch: int = 100, learning_rate: int = 0.00001) -> Cnn:
        test_data_loader, train_data_loader = cls._necessary_data(classes_num, data_set_type)
        model_path = cls._model_path(data_set_type, efficient_net_type.name, classes_num)
        dataset = train_data_loader.dataset

        return Cnn(train_data_loader, test_data_loader,
                   VariantEfficientNet(classes_num, efficient_net_type),
                   model_path, classes=dataset.classes, loss_criterion=loss_criterion, epoch=epoch,
                   learning_rate=learning_rate)

    @classmethod
    def simple_net_instance(cls, data_set_type: DatasetType, classes_num: int, epoch: int = 100,
                            learning_rate: int = 0.00001) -> Cnn:
        test_data_loader, train_data_loader = cls._necessary_data(classes_num, data_set_type)
        path = cls._model_path(data_set_type, 'SIMPLE_CNN_NET', classes_num)
        dataset = train_data_loader.dataset

        return Cnn(train_data_loader, test_data_loader, SimpleNet(classes_num),
                   path, classes=dataset.classes, loss_criterion=loss_criterion, epoch=epoch,
                   learning_rate=learning_rate)

    @classmethod
    def resnet_instance(cls, data_set_type: DatasetType, resnet_type: ResnetType, classes_num: int, epoch: int = 100,
                        learning_rate: int = 0.00001) -> Cnn:
        test_data_loader, train_data_loader = cls._necessary_data(classes_num, data_set_type)
        path = cls._model_path(data_set_type, resnet_type.name, classes_num)
        dataset = train_data_loader.dataset

        return Cnn(train_data_loader, test_data_loader, VariantResnet(classes_num, resnet_type),
                   path, classes=dataset.classes, loss_criterion=loss_criterion, epoch=epoch,
                   learning_rate=learning_rate)

    @classmethod
    def _necessary_data(cls, classes_num, data_set_type):
        train_data_loader = DataLoaderFactory.build(data_set_type, True, classes_num)
        test_data_loader = DataLoaderFactory.build(data_set_type, False, classes_num)
        return test_data_loader, train_data_loader

    @classmethod
    def _model_path(cls, data_set_type: DatasetType, net_type: str, classes_num: int) -> str:
        return './model/' + cls._instance_name(data_set_type, net_type, classes_num)

    @classmethod
    def _instance_name(cls, data_set_type: DatasetType, net_type: str, classes_num: int):
        return net_type + '_' + data_set_type.name + '_' + str(classes_num) + '_' + 'classes'


simple_cnn_cifar10_instance_2_classes = NetBuilder.simple_net_instance(DatasetType.CIFAR10, 2)
simple_cnn_cifar10_instance_3_classes = NetBuilder.simple_net_instance(DatasetType.CIFAR10, 3)
simple_cnn_cifar10_instance_4_classes = NetBuilder.simple_net_instance(DatasetType.CIFAR10, 4)
simple_cnn_cifar10_instance_5_classes = NetBuilder.simple_net_instance(DatasetType.CIFAR10, 5)
simple_cnn_cifar10_instance_10_classes = NetBuilder.simple_net_instance(DatasetType.CIFAR10, 10)

resnet_18_cifar10_instance_2_classes = NetBuilder.resnet_instance(DatasetType.CIFAR10, ResnetType.RESNET18, 2)
resnet_18_cifar10_instance_3_classes = NetBuilder.resnet_instance(DatasetType.CIFAR10, ResnetType.RESNET18, 3)
resnet_18_cifar10_instance_4_classes = NetBuilder.resnet_instance(DatasetType.CIFAR10, ResnetType.RESNET18, 4)
resnet_18_cifar10_instance_5_classes = NetBuilder.resnet_instance(DatasetType.CIFAR10, ResnetType.RESNET18, 5)
resnet_18_cifar10_instance_10_classes = NetBuilder.resnet_instance(DatasetType.CIFAR10, ResnetType.RESNET18, 10)

instance_dict = {
    0: ('SIMPLE_CNN_CIFAR10_INSTANCE_2_CLASSES', simple_cnn_cifar10_instance_2_classes),
    1: ('SIMPLE_CNN_CIFAR10_INSTANCE_3_CLASSES', simple_cnn_cifar10_instance_3_classes),
    2: ('SIMPLE_CNN_CIFAR10_INSTANCE_4_CLASSES', simple_cnn_cifar10_instance_4_classes),
    3: ('SIMPLE_CNN_CIFAR10_INSTANCE_5_CLASSES', simple_cnn_cifar10_instance_5_classes),
    4: ('SIMPLE_CNN_CIFAR10_INSTANCE_10_CLASSES', simple_cnn_cifar10_instance_10_classes),
    5: ('RESNET_CIFAR10_INSTANCE_2_CLASSES', resnet_18_cifar10_instance_2_classes),
    6: ('RESNET_CIFAR10_INSTANCE_3_CLASSES', resnet_18_cifar10_instance_3_classes),
    7: ('RESNET_CIFAR10_INSTANCE_4_CLASSES', resnet_18_cifar10_instance_4_classes),
    8: ('RESNET_CIFAR10_INSTANCE_5_CLASSES', resnet_18_cifar10_instance_5_classes),
    9: ('RESNET_CIFAR10_INSTANCE_10_CLASSES', resnet_18_cifar10_instance_10_classes)
}
