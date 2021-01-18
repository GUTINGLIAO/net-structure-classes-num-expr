from enum import Enum

from net.instance.simple_cnn_cifar10 import *
from net.instance.efficient_net_image_net import *


class Net(Enum):
    """Enum of different type of net instance.

    The name of Enum is the name of the net and the value of Enum is the instance of the net.
    The name is combined by the net structure, the data set and the number of classes waiting to be classified
    """

    SIMPLE_CNN_CIFAR10_2_CLASSES = simple_cnn_cifar10_instance_2_classes
    SIMPLE_CNN_CIFAR10_3_CLASSES = simple_cnn_cifar10_instance_3_classes
    SIMPLE_CNN_CIFAR10_4_CLASSES = simple_cnn_cifar10_instance_4_classes
    SIMPLE_CNN_CIFAR10_5_CLASSES = simple_cnn_cifar10_instance_5_classes
    SIMPLE_CNN_CIFAR10_10_CLASSES = simple_cnn_cifar10_instance_10_classes

    EFFICIENT_NET_IMAGE_NET_2_CLASSES = efficient_net_b0_image_net_instance_2_classes
    EFFICIENT_NET_IMAGE_NET_3_CLASSES = 6
    EFFICIENT_NET_IMAGE_NET_4_CLASSES = 7
    EFFICIENT_NET_IMAGE_NET_5_CLASSES = 8
    EFFICIENT_NET_IMAGE_NET_10_CLASSES = 9
    EFFICIENT_NET_IMAGE_NET_50_CLASSES = 10
    EFFICIENT_NET_IMAGE_NET_100_CLASSES = 11