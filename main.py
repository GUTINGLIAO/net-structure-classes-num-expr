from net.cnn import Cnn
from net.instance import resnet_18_cifar10_instance_2_classes

if __name__ == '__main__':
    net: Cnn = resnet_18_cifar10_instance_2_classes
    net.train()
