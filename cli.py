import click

from net.cnn import Cnn
from net.instance import NetType, NetBuilder, instance_dict
from net.structure.resnet import ResnetType
from net.structure.efficient_net import EfficientNetType
from net.instance.data import DatasetType


# TODO 增加其余配置，epoch，lr等
@click.group()
def cli():
    pass


@click.command()
def resource():
    print('net')
    for _type in NetType:
        print('name: %s, index: %d' % (_type.name, _type.value))

    print('resnet_type')
    for _type in ResnetType:
        print('name: %s, index: %d' % (_type.name, _type.value))

    print('efficient_net_type')
    for _type in EfficientNetType:
        print('name: %s, index: %d' % (_type.name, _type.value))

    print('operation')
    print('operation: train')
    print('operation: test')

    print('dataset')
    for _type in DatasetType:
        print('name: %s, index: %d' % (_type.name, _type.value))

    print('instance')
    for key, value in instance_dict.items():
        print('name: %s, index: %d' % (value[0], key))


# FIXME 是否有更好的实现
@click.command()
@click.option('--net', default=-1, help='Appoint net.')
@click.option('--type', default=-1, help='Appoint specific net type if neccessary')
@click.option('--classes_num', help='Appoint the num of classes.')
@click.option('--dataset', help='Appoint the type of dataset.')
@click.option('--instance', default=-1, help='Appoint existed instance.')
@click.option('--lr', default=0.00001, help='Appoint learning rate, default is 0.00001.')
@click.option('--epoch', default=100, help='Appoint learning epoch, default is 100.')
def train(net: int, type: int, dataset: int, classes_num: int, instance: int, lr: int, epoch: int):
    if net == -1 and instance == -1:
        raise ValueError('Please create a new instance or appoint a existed instance.')
    if net == -1:
        net_instance: Cnn = instance_dict[instance][1]
        net_instance.train()
        return
    if net == NetType.SIMPLE_NET.value:
        NetBuilder.simple_net_instance(classes_num=classes_num, data_set_type=DatasetType.convert(dataset),
                                       learning_rate=lr, epoch=epoch)
        return

    if type == -1:
        raise ValueError('Please appoint a specific net type')

    if net == NetType.RESNET.value:
        NetBuilder.resnet_instance(classes_num=classes_num, data_set_type=DatasetType.convert(dataset),
                                   resnet_type=ResnetType.convert(type), learning_rate=lr, epoch=epoch)
        return

    if net == NetType.EFFICIENT_NET.value:
        NetBuilder.efficient_net_instance(classes_num=classes_num, data_set_type=DatasetType.convert(dataset),
                                          efficient_net_type=EfficientNetType.convert(type), learning_rate=lr,
                                          epoch=epoch)


# FIXME 是否有更好的实现
@click.command()
@click.option('--net', default=-1, help='Appoint the type of net.')
@click.option('--classes_num', help='Appoint the num of classes.')
@click.option('--dataset', help='Appoint the type of dataset.')
@click.option('--instance', default=-1, help='Appoint existed instance.')
def test(net: int, dataset: int, classes_num: int, instance: int):
    if net == -1 and instance == -1:
        raise ValueError('Please create a new instance or appoint a existed instance.')
    if net == -1:
        net_instance: Cnn = instance_dict[instance][1]
        net_instance.test()
        return
    if net == NetType.SIMPLE_NET.value:
        NetBuilder.simple_net_instance(classes_num=classes_num, data_set_type=DatasetType.convert(dataset))


cli.add_command(resource)
cli.add_command(train)
cli.add_command(test)

if __name__ == '__main__':
    cli()
