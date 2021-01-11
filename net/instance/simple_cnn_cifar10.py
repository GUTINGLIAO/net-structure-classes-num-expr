from net.instance.common_configuration import loss_criterion
from net.cnn import Cnn
from net.instance.data import DataLoaderFactory, DatasetType
from net.structure.simple import SimpleNet

train_data_loader_2_classes = DataLoaderFactory.build(DatasetType.CIFAR10, True, 2)
test_data_loader_2_classes = DataLoaderFactory.build(DatasetType.CIFAR10, False, 2)
classes_2 = ('plane', 'car')
simple_cnn_cifar10_instance_2_classes = Cnn(train_data_loader_2_classes,
                                            test_data_loader_2_classes, SimpleNet(2),
                                            './model/simple_cnn_cifar10_2_classes',
                                            classes_2, loss_criterion, epoch=100)

train_data_loader_3_classes = DataLoaderFactory.build(DatasetType.CIFAR10, True, 3)
test_data_loader_3_classes = DataLoaderFactory.build(DatasetType.CIFAR10, False, 3)
classes_3 = ('plane', 'car', 'bird')
simple_cnn_cifar10_instance_3_classes = Cnn(train_data_loader_3_classes,
                                            test_data_loader_3_classes, SimpleNet(3),
                                            './model/simple_cnn_cifar10_3_classes',
                                            classes_3, loss_criterion, epoch=100)

train_data_loader_4_classes = DataLoaderFactory.build(DatasetType.CIFAR10, True, 4)
test_data_loader_4_classes = DataLoaderFactory.build(DatasetType.CIFAR10, False, 4)
classes_4 = ('plane', 'car', 'bird', 'cat')
simple_cnn_cifar10_instance_4_classes = Cnn(train_data_loader_4_classes,
                                            test_data_loader_4_classes, SimpleNet(4),
                                            './model/simple_cnn_cifar10_4_classes',
                                            classes_4, loss_criterion, epoch=100)

train_data_loader_5_classes = DataLoaderFactory.build(DatasetType.CIFAR10, True, 5)
test_data_loader_5_classes = DataLoaderFactory.build(DatasetType.CIFAR10, False, 5)
classes_5 = ('plane', 'car', 'bird', 'cat', 'deer')
simple_cnn_cifar10_instance_5_classes = Cnn(train_data_loader_5_classes,
                                            test_data_loader_5_classes, SimpleNet(5),
                                            './model/simple_cnn_cifar10_5_classes',
                                            classes_5, loss_criterion, epoch=100)

train_data_loader_10_classes = DataLoaderFactory.build(DatasetType.CIFAR10, True, 10)
test_data_loader_10_classes = DataLoaderFactory.build(DatasetType.CIFAR10, False, 10)
classes_10 = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
simple_cnn_cifar10_instance_10_classes = Cnn(train_data_loader_10_classes,
                                             test_data_loader_10_classes, SimpleNet(10),
                                             './model/simple_cnn_cifar10_10_classes',
                                             classes_10, loss_criterion, epoch=100)
