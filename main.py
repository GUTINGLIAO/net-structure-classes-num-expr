from net.cnn import Cnn, device
from net.instance import simple_cnn_cifar10_instance_10_classes

if __name__ == '__main__':
    print(device)
    net: Cnn = simple_cnn_cifar10_instance_10_classes
    net.test()


