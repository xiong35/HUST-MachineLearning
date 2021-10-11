# encoding=utf8
import numpy as np


class kNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.train_feature = None
        self.train_label = None

    def fit(self, feature, label):
        self.train_feature = feature
        self.train_label = label

    def predict(self, feature):
        result = []

        for feat in feature:
            diff = self.train_feature - feat
            sq_diff = diff ** 2
            dist = sq_diff.sum(axis=1) ** 0.5

            dist_index = dist.argsort()
            sorted_labels = self.train_label[dist_index]

            class_count = {}
            for i in range(self.k):
                label = sorted_labels[i]
                class_count[label] = class_count.get(label, 0) + 1
            sorted_class_count = sorted(
                class_count.items(), key=lambda x: x[1], reverse=True)
            result.append(sorted_class_count[0][0])

        return result
