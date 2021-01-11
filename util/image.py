# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
#
#
# # functions to show an image
# from net.instance.simple_cnn_cifar10 import train_data_loader, classes_10
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(train_data_loader)
# images, labels = dataiter.next()
#
# if __name__ == '__main__':
#
#     for data in train_data_loader.dataset:
#         imgs, labs = data
#         print(1)
#
#     # show images
#     imshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(' '.join('%5s' % classes_10[labels[j]] for j in range(4)))