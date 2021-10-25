# encoding=utf8
import pandas as pd
from sklearn.svm import LinearSVC


def linearsvc_predict(train_data, train_label, test_data):

    # 获取训练数据
    train_data = pd.read_csv('./step1/train_data.csv')
    # 获取训练标签
    train_label = pd.read_csv('./step1/train_label.csv')
    train_label = train_label['target']
    # 获取测试数据
    test_data = pd.read_csv('./step1/test_data.csv')

    svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.00001, C=0.71, multi_class='crammer_singer',
                    fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=2000)

    svc.fit(train_data, train_label)

    predict = svc.predict(test_data)
    return predict
