
import numpy as np


# 想知道为什么可以按照题目描述里的指示看看这个数据集长啥样
def alcohol_mean(data):
    '''
    返回红酒数据中红酒的酒精平均含量
    :param data: 红酒数据对象
    :return: 酒精平均含量，类型为float
    '''
    return data.data[:, 0].mean()
