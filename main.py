#%%

import paddle
print(paddle.__version__)

#%%

# 下载数据集 
!wget -O OCR_Dataset.zip https://bj.bcebos.com/v1/ai-studio-online/c91f50ef72de43b090298a38281e9c59a2d741eadd334f1cba7c710c5496e342?responseContentDisposition=attachment%3B%20filename%3DOCR_Dataset.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-10-27T09%3A50%3A21Z%2F-1%2F%2Fddc4aebed803af6c57dac46abba42d207961b78e7bc81744e8388395979b66fa

#%%

# 解压数据集
!unzip OCR_Dataset.zip -d data/

#%%

import os

import PIL.Image as Image
import numpy as np
from paddle.io import Dataset

# 图片信息配置 - 通道数、高度、宽度
IMAGE_SHAPE_C = 3
IMAGE_SHAPE_H = 30
IMAGE_SHAPE_W = 70
# 数据集图片中标签长度最大值设置 - 因图片中均为4个字符，故该处填写为4即可
LABEL_MAX_LEN = 4


class Reader(Dataset):
    def __init__(self, data_path: str, is_val: bool = False):
        """
        数据读取Reader
        :param data_path: Dataset路径
        :param is_val: 是否为验证集
        """
        super().__init__()
        self.data_path = data_path
        # 读取Label字典
        with open(os.path.join(self.data_path, "label_dict.txt"), "r", encoding="utf-8") as f:
            self.info = eval(f.read())
        # 获取文件名列表
        self.img_paths = [img_name for img_name in self.info]
        # 将数据集后1024张图片设置为验证集，当is_val为真时img_path切换为后1024张
        self.img_paths = self.img_paths[-1024:] if is_val else self.img_paths[:-1024]

    def __getitem__(self, index):
        # 获取第index个文件的文件名以及其所在路径
        file_name = self.img_paths[index]
        file_path = os.path.join(self.data_path, file_name)
        # 捕获异常 - 在发生异常时终止训练
        try:
            # 使用Pillow来读取图像数据
            img = Image.open(file_path)
            # 转为Numpy的array格式并整体除以255进行归一化
            img = np.array(img, dtype="float32").reshape((IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)) / 255
        except Exception as e:
            raise Exception(file_name + "\t文件打开失败，请检查路径是否准确以及图像文件完整性，报错信息如下:\n" + str(e))
        # 读取该图像文件对应的Label字符串，并进行处理
        label = self.info[file_name]
        label = list(label)
        # 将label转化为Numpy的array格式
        label = np.array(label, dtype="int32")

        return img, label

    def __len__(self):
        # 返回每个Epoch中图片数量
        return len(self.img_paths)

