from net import Net
from net.cnn import Cnn

net: Cnn = Net.EFFICIENT_NET_IMAGE_NET_2_CLASSES.value

if __name__ == '__main__':
    net.train()
