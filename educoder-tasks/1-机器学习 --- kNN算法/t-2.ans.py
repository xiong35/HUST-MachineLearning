from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classification(train_feature, train_label, test_feature):

    scaler = StandardScaler()
    train_feature = scaler.fit_transform(train_feature)

    classifier = KNeighborsClassifier(5)
    classifier.fit(train_feature, train_label)

    return classifier.predict(scaler.transform(test_feature))
