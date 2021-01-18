from net import Net, Cnn

net: Cnn = Net.SIMPLE_CNN_CIFAR10_2_CLASSES.value

if __name__ == '__main__':
    net.train()