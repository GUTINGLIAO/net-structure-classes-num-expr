from net.cnn import Cnn
from net.instance import simple_cnn_cifar10_instance_2_classes

if __name__ == '__main__':
    net: Cnn = simple_cnn_cifar10_instance_2_classes
    net.test()
