import os
from collections import Counter
import pandas as pd
import numpy as np
import jieba as jb


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


top_words = 2000
epoch_num = 5
batch_size = 64
max_review_length = 500


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()


def load_data():
    if os.path.isfile('./__data__/xtr_LSTM.npy') and\
            os.path.isfile('./__data__/xte_LSTM.npy') and \
            os.path.isfile('./__data__/ytr_LSTM.npy') and \
            os.path.isfile('./__data__/yte_LSTM.npy'):
        xtr = np.load('./__data__/xtr_LSTM.npy')
        xte = np.load('./__data__/xte_LSTM.npy')
        ytr = np.load('./__data__/ytr_LSTM.npy')
        yte = np.load('./__data__/yte_LSTM.npy')
    else:
        path = '../data/train.csv'
        data = pd.read_csv(path)[:1000]
        data.fillna('', inplace=True)

        stopword_set = set()
        with open("../data/stopwords.txt", "r", encoding="utf-8") as fr:
            words = fr.readline()
            while words:
                stopword_set.add(words.replace("\n", ""))
                words = fr.readline()

        x = []
        y = []

        for i in range(len(data)):
            line = data['content'][i] + data['comment_all'][i]
            words = jb.lcut(line)
            words = [w for w in words if w not in stopword_set]
            x.append(words)
            y.append(data['label'][i])

        train_size = int(0.8 * len(y))

        xtr = x[:train_size]
        xte = x[train_size:]
        ytr = y[:train_size]
        yte = y[train_size:]

        cnt = Counter()
        for x in xtr:
            for word in x:
                cnt[word] += 1

        most_common = cnt.most_common(top_words + 1)
        word_bank = {}
        id_num = 1
        for word, freq in most_common:
            word_bank[word] = id_num
            id_num += 1

        for news in xtr:
            i = 0
            while i < len(news):
                if news[i] in word_bank:
                    news[i] = word_bank[news[i]]
                    i += 1
                else:
                    del news[i]

        for news in xte:
            i = 0
            while i < len(news):
                if news[i] in word_bank:
                    news[i] = word_bank[news[i]]
                    i += 1
                else:
                    del news[i]

        xtr = sequence.pad_sequences(xtr, maxlen=max_review_length)
        xte = sequence.pad_sequences(xte, maxlen=max_review_length)
        ytr = np.array(ytr)
        yte = np.array(yte)

    np.save('./__data__/xtr_LSTM', xtr)
    np.save('./__data__/xte_LSTM', xte)
    np.save('./__data__/ytr_LSTM', ytr)
    np.save('./__data__/yte_LSTM', yte)

    return xtr, xte, ytr, yte


X_train, X_test, y_train, y_test = load_data()

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+2, embedding_vecor_length,
                    input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=5, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy= %.2f%%" % (scores[1]*100))

# Draw the confusion matrix
y_pred = model.predict_classes(X_test)
plot_cmat(y_test, y_pred)
