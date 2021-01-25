from net.cnn import Cnn
from net.instance.data import DataLoaderFactory, DatasetType
from net.instance.common_configuration import loss_criterion
from net.structure.efficient_net import VariantEfficientNet, EfficientNetType

__all__ = ['NetBuilder', 'simple_cnn_cifar10_instance_2_classes',
           'simple_cnn_cifar10_instance_3_classes', 'simple_cnn_cifar10_instance_4_classes',
           'simple_cnn_cifar10_instance_5_classes', 'simple_cnn_cifar10_instance_10_classes']

from net.structure.simple import SimpleNet


class NetBuilder():

    @classmethod
    def efficient_net_instance(cls, data_set_type: DatasetType, efficient_net_type: EfficientNetType,
                               classes_num: int, epoch: int = 100, learning_rate: int = 0.00001) -> Cnn:
        test_data_loader, train_data_loader = cls._necessary_data(classes_num, data_set_type)
        model_path = cls._model_path(data_set_type, 'efficient_net' + efficient_net_type.name, classes_num)
        dataset = train_data_loader.dataset

        return Cnn(train_data_loader, test_data_loader,
                   VariantEfficientNet(classes_num, efficient_net_type),
                   model_path, classes=dataset.classes, loss_criterion=loss_criterion, epoch=epoch,
                   learning_rate=learning_rate)

    @classmethod
    def simple_net_instance(cls, data_set_type: DatasetType, classes_num: int, epoch: int = 100,
                            learning_rate: int = 0.00001) -> Cnn:
        test_data_loader, train_data_loader = cls._necessary_data(classes_num, data_set_type)
        path = cls._model_path(data_set_type, 'simple_cnn_net', classes_num)
        dataset = train_data_loader.dataset

        return Cnn(train_data_loader, test_data_loader, SimpleNet(classes_num),
                   path, classes = dataset.classes, loss_criterion=loss_criterion, epoch=epoch, learning_rate=learning_rate)

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
