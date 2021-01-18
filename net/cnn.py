import os

import torch
from torch.nn import Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from util.time import now


class Cnn:
    """A complete data structure for using a cnn.

    This class contains all necessary configuration to train and test a cnn.

    :param train_data_loader: A data loader that provides training data.
    :param test_data_loader: A data loader that provides testing data.
    :param net_structure: neural net structure without pretrained.
    :param learning_rate: this param decides how fast can the net approach the extreme point.
    :param epoch: this param decides how many time the net is trained.
    :param loss_criterion: this param provide a function to calculate the loss.
    :param optimizer: this param is used for gradient calculation and back propagation.
    :param path: to appoint a path to save trained model.
    :param classes: to appoint classes that you want to classify.
    """

    # This project has included three dataset, I.e, CIFAR10, MINISET, IMAGENET.
    train_data_loader: DataLoader = ...
    test_data_loader: DataLoader = ...

    # This project has included three neural net, i.e., Simple Net, Resnet, Efficientnet.
    net_structure: Module = ...

    # In current project, we use CrossEntropyLoss for all nets.
    loss_criterion: Module = ...

    # In current project, we use SGD for all nets.
    optimizer: Optimizer = ...

    # Normally, this param will be set between [0.00001, 0.1].
    # Undersize value will prolong the training time.
    # Oversize value will make the net hard to converge.
    learning_rate: int

    # This param do not have a common value because it is strongly related to the number of net layers.
    # In my personal experience, a cnn with two convolutional layers requires about 200 epochs to converge.
    epoch: int

    path: str

    classes: tuple

    def __init__(self, train_data_loader: DataLoader, test_data_loader: DataLoader,
                 net_structure: Module, path: str, classes: tuple, loss_criterion: Module,
                 learning_rate: int = 0.00001, epoch: int = 10):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.net_structure = net_structure
        self.path = path
        self.classes = classes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss_criterion = loss_criterion
        self.optimizer = SGD(self.net_structure.parameters(), learning_rate, momentum=0.9)

    def train(self):

        if os.path.exists(self.path):
            print('model has existed')
            self.net_structure.load_state_dict(torch.load(self.path))

        for epoch in range(self.epoch):  # loop over the dataset multiple times

            print('start epoch %d at %s' % (epoch + 1, now()))
            running_loss = 0.0
            for i, data in enumerate(self.train_data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs: torch.tensor = self.net_structure(inputs)
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

            print('finish epoch %d at %s' % (epoch + 1, now()))

            torch.save(self.net_structure.state_dict(), self.path)

            print('model is saved')

    def test(self):

        if not os.path.exists(self.path):
            raise Exception('no trained model exists')
        self.net_structure.load_state_dict(torch.load(self.path))

        class_correct = list(0. for _ in range(self.classes.__len__()))
        class_total = list(0. for _ in range(self.classes.__len__()))

        with torch.no_grad():
            for data in self.test_data_loader:
                images, labels = data
                outputs = self.net_structure(images)
                _, predicted = torch.max(outputs, 1)

                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i in range(self.classes.__len__()):
                print('Accuracy of %5s : %2d %%' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i]))
