# encoding=utf8
import os
import pandas as pd
import numpy as np
from sklearn.linear_model.perceptron import Perceptron
from sklearn.preprocessing import StandardScaler


# 获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

# 标准化数据
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

clf = Perceptron()
clf.fit(train_data, train_label)
pred = clf.predict(scaler.transform(test_data))

result = np.where(pred > 0.5, 1, 0)

df = pd.DataFrame(result, columns=["result"])
df.to_csv('./step2/result.csv')
