import click

from net.cnn import Cnn
from net.instance import NetType, NetBuilder, instance_dict
from net.instance.data import DatasetType


@click.group()
def cli():
    pass


@click.command()
def resource():
    print('net type')
    for _type in NetType:
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
@click.option('--net', default=-1, help='Appoint the type of net.')
@click.option('--classes_num', help='Appoint the num of classes.')
@click.option('--dataset', help='Appoint the type of dataset.')
@click.option('--instance', default=-1, help='Appoint existed instance.')
def train(net: int, dataset: int, classes_num: int, instance: int):
    if net == -1 and instance == -1:
        raise ValueError('Please create a new instance or appoint a existed instance.')
    if net == -1:
        net_instance: Cnn = instance_dict[instance][1]
        net_instance.train()
        return
    if net == NetType.SIMPLE_NET.value:
        NetBuilder.simple_net_instance(classes_num=classes_num, data_set_type=DatasetType.convert(dataset))


@click.command()
@click.option('--model', required=True, type=int, help='Appoint the model that you want to test.')
def test():
    pass


cli.add_command(resource)
cli.add_command(train)
cli.add_command(test)

if __name__ == '__main__':
    cli()
