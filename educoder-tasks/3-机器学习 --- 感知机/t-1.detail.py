# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.]*data.shape[1])       # 将 weight 置为全 1 的数组
        self.b = np.array([1.])                     # 将 bias 置为 1

        length = data.shape[0]  # 迭代数据时会多次用到数据集长度

        for i in range(self.max_iter):  # 如果超出最大迭代次数就停止训练
            has_error = False   # 如果后续发现一次训练没有错误也停止训练

            for i in range(length):     # 迭代训练数据
                x = data[i]             # x 为当前数据
                # 即 x_1*w_1 + x_2*w_2 + ... + x_i*w_i + b, 用向量乘法会简介一些
                y = x.dot(self.w) + self.b

                res = 1 if y > 0 else - 1       # 算出来的 y 接近 1 就相当于预测结果是 1, 接近 -1 则预测 -1
                if res == label[i]:             # 如果预测对了就不干事
                    continue

                else:                           # 如果预测错了
                    has_error = True
                    # 按题目中给的公式更新 w 和 b
                    self.w -= self.lr * res * x
                    self.b -= self.lr * res
            if not has_error:
                break

        #********* End *********#
    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''

        predict = []
        for x in data:
            y = self.w.dot(x) + self.b
            res = 1 if y > 0 else - 1
            predict.append(res)

        return predict
