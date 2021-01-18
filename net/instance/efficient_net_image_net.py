from net.cnn import Cnn
from net.instance.data import DataLoaderFactory, DatasetType
from net.structure.efficient_net import VariantEfficientNet, EfficientNetType

__all__ = ['efficient_net_b0_image_net_instance_2_classes']

train_data_loader_2_classes = DataLoaderFactory.build(DatasetType.IMAGENET, True, 2)
test_data_loader_2_classes = DataLoaderFactory.build(DatasetType.IMAGENET, False, 2)
classes_2 = ('plane', 'car')
efficient_net_b0_image_net_instance_2_classes = Cnn(train_data_loader_2_classes,
                                                    test_data_loader_2_classes,
                                                    VariantEfficientNet(2, EfficientNetType.B0),
                                                    './model/efficient_net_b0_image_net_instance_2_classes',
                                                    classes_2, loss_criterion, epoch=100)
