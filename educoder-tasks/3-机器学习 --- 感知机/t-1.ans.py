# encoding=utf8
import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])

        length = data.shape[0]
        for i in range(self.max_iter):
            has_error = False

            for i in range(length):
                x = data[i]
                y = x.dot(self.w) + self.b

                res = 1 if y > 0 else - 1
                if res == label[i]:
                    continue

                else:
                    has_error = True
                    self.w -= self.lr * res * x
                    self.b -= self.lr * res
            if not has_error:
                break

    def predict(self, data):
        predict = []
        for x in data:
            y = self.w.dot(x) + self.b
            res = 1 if y > 0 else - 1
            predict.append(res)

        return predict
