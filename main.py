#%%

import paddle
import paddle.vision as vision
import paddle.text as text

paddle.__version__

#%%

print('视觉相关数据集：', paddle.vision.datasets.__all__)
print('自然语言相关数据集：', paddle.text.__all__)

#%%

from paddle.vision.transforms import ToTensor
# 训练数据集
train_dataset = vision.datasets.MNIST(mode='train', transform=ToTensor())

# 验证数据集
val_dataset = vision.datasets.MNIST(mode='test', transform=ToTensor())

