import argparse

from net.cnn import Cnn
from net.instance import NetFactory
from net.type import NetType

# TODO use docopt
net_type_dict = {
    1: NetType.SIMPLE_CNN_CIFAR10_2_CLASSES,
    2: NetType.SIMPLE_CNN_CIFAR10_3_CLASSES,
    3: NetType.SIMPLE_CNN_CIFAR10_4_CLASSES,
    4: NetType.SIMPLE_CNN_CIFAR10_5_CLASSES,
    5: NetType.SIMPLE_CNN_CIFAR10_10_CLASSES,
}

parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True, type=int,
                    help='Appoint the type of net that you want to get. Examples: %s' % str(net_type_dict))
parser.add_argument('--oper', required=True, type=str,
                    help='Input train means you want to train the net, test means you want to test the net')

argv = parser.parse_args()

net_type = net_type_dict.get(argv.net)

net:Cnn = NetFactory.build(net_type)

if argv.oper == 'train':
    net.train()

if argv.oper == 'test':
    net.test()
