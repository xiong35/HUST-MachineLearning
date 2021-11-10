from numpy.core.numeric import False_
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import jieba as jb
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np


def load_array(path, stopword_set=None, word_dict=None, is_test=False):
    print("### load_array ###")
    df = pd.read_csv(path)
    lines = []
    label = []

    for i in range(0, df.size):
        try:
            lines.append(df["content"][i] + df["comment_all"][i])
            if not is_test:
                label.append(df["label"][i])
        except:
            pass

    data = word_break(lines)
    data, stopword_set = cleanup_stopword(data, stopword_set)
    data, word_dict = remove_spacial_word(data, word_dict)

    if is_test:
        label = None

    return data, label, stopword_set, word_dict


def word_break(lines):
    print("### word_break ###")

    return [jb.lcut(l) for l in lines]


def cleanup_stopword(dataset, stopword_set):
    print("### cleanup_stopword ###")
    if stopword_set is None:
        stopword_set = set()
        with open("./data/stopwords.txt", "r", encoding="utf-8") as fr:
            words = fr.readline()
            while words:
                stopword_set.add(words.replace("\n", ""))
                words = fr.readline()

    return [[w for w in data if w not in stopword_set] for data in dataset], stopword_set


def remove_spacial_word(dataset, word_dict):
    print("### remove_spacial_word ###")
    if word_dict is None:
        word_dict = {}
    for data in dataset:
        for word in data:
            word_dict[word] = word_dict.get(word, 1) + 1

    for i in range(0, len(dataset)):
        new_data = []
        for word in dataset[i]:
            if(word_dict.get(word, 0) > 1):
                new_data.append(word)
        dataset[i] = new_data

    return dataset, word_dict


def array2vect(x_train, y_train, x_test, y_test):
    print("### array2vect ###")
    vector_dimension = 300

    x = []

    total = []
    total.extend(x_train)
    total.extend(x_test)

    for i, data in enumerate(total):
        x.append(TaggedDocument(data, ['Text' + '_%s' % str(i)]))

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count,
                     epochs=10)

    train_size = len(x_train)
    test_size = len(x_test)

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y_train[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        try:
            test_labels[j] = y_test[j]
        except:
            pass
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def get_vects(x_train, y_train, x_test, y_test):
    print("### get_vects ###")
    if not os.path.isfile('./__data__/xtr.npy') or \
            not os.path.isfile('./__data__/xte.npy') or \
            not os.path.isfile('./__data__/ytr.npy') or \
            not os.path.isfile('./__data__/yte.npy'):
        xtr, xte, ytr, yte = array2vect(x_train, y_train, x_test, y_test)
        np.save('./__data__/xtr', xtr)
        np.save('./__data__/xte', xte)
        np.save('./__data__/ytr', ytr)
        np.save('./__data__/yte', yte)

    xtr = np.load('./__data__/xtr.npy')
    xte = np.load('./__data__/xte.npy')
    ytr = np.load('./__data__/ytr.npy')
    yte = np.load('./__data__/yte.npy')

    return xtr, xte, ytr, yte


def plot_cmat(yte, ypred):
    print("### plot_cmat ###")
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()


def main():
    print("### main ###")
    x_train, y_train, stopword_set, word_dict = load_array(
        "./data/train.csv")

    # x_test, _, _, _ = load_array("./data/test.csv", stopword_set, word_dict)

    train_size = int(0.8 * len(x_train))

    xtr, xte, ytr, yte = get_vects(
        x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:])

    clf = SVC()
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)
    print(y_pred[:100])
    m = yte.shape[0]
    n = (yte != y_pred).sum()
    print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

    # Draw the confusion matrix
    plot_cmat(yte, y_pred)


main()
