from net.type import NetType


class NetFactory:

    @classmethod
    def build(cls, net_type: NetType):
        """To create a Cnn instance.

        :param net_type: contain the net structure type and the data set type.
        :return: a Cnn instance.
        """
        return net_type.value