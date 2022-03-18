#%%

import paddle
import paddle.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

print(paddle.__version__)

#%%

import paddle.vision.transforms as T
transform = T.Compose([T.Transpose((2, 0, 1))])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
x_train = np.zeros((50000, 3, 32, 32))
y_train = np.zeros((50000, 1), dtype='int32')

for i in range(len(cifar10_train)):
    train_image, train_label = cifar10_train[i]
    
    # normalize the data
    x_train[i,:, :, :] = train_image / 255.
    y_train[i, 0] = train_label

y_train = np.squeeze(y_train)

print(x_train.shape)
print(y_train.shape)

#%%

cifar10_test = paddle.vision.datasets.cifar.Cifar10(mode='test', transform=transform)
x_test = np.zeros((10000, 3, 32, 32), dtype='float32')
y_test = np.zeros((10000, 1), dtype='int64')

for i in range(len(cifar10_test)):
    test_image, test_label = cifar10_test[i]
   
    # normalize the data
    x_test[i,:, :, :] = test_image / 255.
    y_test[i, 0] = test_label

y_test = np.squeeze(y_test)

print(x_test.shape)
print(y_test.shape)

