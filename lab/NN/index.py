from keras.utils import to_categorical
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import jieba as jb
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import scikitplot.plotters as skplt
import os


def load_array(path, stopword_set=None, word_dict=None, is_test=False):
    print("### load_array ###")
    df = pd.read_csv(path)
    df.fillna('', inplace=True)

    lines = []
    label = []

    for i in range(0, df.shape[0]):
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
        with open("../data/stopwords.txt", "r", encoding="utf-8") as fr:
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
        if y_test is not None:
            test_labels[j] = y_test[j]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def plot_cmat(yte, ypred):
    print("### plot_cmat ###")
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()


def baseline_model():
    '''Neural network with 3 hidden layers'''
    model = Sequential()

    model.add(Dense(256, input_dim=300, activation='relu',
                    kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(3, activation="softmax", kernel_initializer='normal'))

    # gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # configure the learning process of the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def main():
    print("### main ###")
    if not os.path.isfile('./__data__/xtr.npy') or \
            not os.path.isfile('./__data__/xte.npy') or \
            not os.path.isfile('./__data__/ytr.npy') or \
            not os.path.isfile('./__data__/yte.npy'):
        x_train, y_train, stopword_set, word_dict = load_array(
            "../data/train.csv")
        x_test, _, _, _ = load_array(
            "../data/test.csv", stopword_set, word_dict, is_test=True)
        xtr, xte, ytr, yte = array2vect(x_train, y_train, x_test, None)
        np.save('./__data__/xtr', xtr)
        np.save('./__data__/xte', xte)
        np.save('./__data__/ytr', ytr)
        np.save('./__data__/yte', yte)
    else:
        xtr = np.load('./__data__/xtr.npy')
        xte = np.load('./__data__/xte.npy')
        ytr = np.load('./__data__/ytr.npy')
        yte = np.load('./__data__/yte.npy')

    encoded_y = []
    for y in ytr:
        encoded_y_item = [0, 0, 0]
        encoded_y_item[int(y)+1] = 1
        encoded_y.append(encoded_y_item)
    ytr = np.array(encoded_y)

    print("### data loaded ###")

    # Train the model
    model = baseline_model()
    model.summary()

    estimator = model.fit(xtr, ytr, epochs=20, batch_size=64)
    print("Model Trained!")

    probabs = model.predict_proba(xte)
    print("probabs.shape", probabs.shape)
    y_pred = np.argmax(probabs, axis=1) - 1

    np.savetxt('./result.txt', y_pred, fmt="%d", delimiter=" ")


main()
