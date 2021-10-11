from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''

    # 实例化一个 StandardScaler 对象
    scaler = StandardScaler()
    # scaler.fit_transform 会将数据进行标准化, 同时记录数据的均值和方差以便对后续测试数据执行同样的标准化
    std_train_feature = scaler.fit_transform(train_feature)

    # 实例化一个KNN分类器
    classifier = KNeighborsClassifier()
    # 使用标准化后的数据训练他
    classifier.fit(std_train_feature, train_label)

    # 返回(使用(训练过的分类器)预测(标准化后的数据)的结果)
    return classifier.predict(scaler.transform(test_feature))
