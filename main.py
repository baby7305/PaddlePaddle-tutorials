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

#%%

from paddle.io import Dataset


class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()

        if mode == 'train':
            self.data = [
                ['traindata1', 'label1'],
                ['traindata2', 'label2'],
                ['traindata3', 'label3'],
                ['traindata4', 'label4'],
            ]
        else:
            self.data = [
                ['testdata1', 'label1'],
                ['testdata2', 'label2'],
                ['testdata3', 'label3'],
                ['testdata4', 'label4'],
            ]
    
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)

# 测试定义的数据集
train_dataset_2 = MyDataset(mode='train')
val_dataset_2 = MyDataset(mode='test')

print('=============train dataset=============')
for data, label in train_dataset_2:
    print(data, label)

print('=============evaluation dataset=============')
for data, label in val_dataset_2:
    print(data, label)

#%%

from paddle.vision.transforms import Compose, Resize, ColorJitter

# 定义想要使用那些数据增强方式，这里用到了随机调整亮度、对比度和饱和度，改变图片大小
transform = Compose([ColorJitter(), Resize(size=100)])

# 通过transform参数传递定义好的数据增项方法即可完成对自带数据集的应用
train_dataset_3 = vision.datasets.MNIST(mode='train', transform=transform)

