import os
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd

import jieba as jb

stopword_dict = set()
with open("../data/stopwords.txt", "r", encoding="utf-8") as fr:
    words = fr.readline()
    while words:
        stopword_dict.add(words.replace("\n", ""))
        words = fr.readline()


def str2words(string):
    ret_arr = []
    cut = jb.lcut(string)
    for w in cut:
        if w not in stopword_dict:
            ret_arr.append(w)
    return ret_arr


def dataset2array():
    train_data_origin = pd.read_csv("../data/train.csv")
    train_data = []
    train_label = []

    for i in range(0, train_data_origin.size):
        try:
            train_data.append(str2words(
                train_data_origin["content"][i]) + str2words(train_data_origin["comment_all"][i]))
            train_label.append(train_data_origin["label"][i])
        except:
            pass

    return remove_spacial_words(train_data), train_label


def remove_spacial_words(dataset):
    word_dict = {}
    for data in dataset:
        for word in data:
            word_dict[word] = word_dict.get(word, 1) + 1

    for i in range(0, len(dataset)):
        new_data = []
        for word in dataset[i]:
            if(word_dict[word] > 1):
                new_data.append(word)
        dataset[i] = new_data

    return dataset


def array2tagged_document(dataset):
    sentences = []
    for i, data in enumerate(dataset):
        sentences.append(TaggedDocument(data, ['Text' + '_%s' % str(i)]))

    return sentences


def tagged2vect(x, y, vector_dimension=300):
    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count,
                     epochs=10)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def get_vects():
    if not os.path.isfile('./xtr.npy') or \
            not os.path.isfile('./xte.npy') or \
            not os.path.isfile('./ytr.npy') or \
            not os.path.isfile('./yte.npy'):
        td, tl = dataset2array()
        tagged = array2tagged_document(td)
        xtr, xte, ytr, yte = tagged2vect(tagged, tl)
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)

    xtr = np.load('./xtr.npy')
    xte = np.load('./xte.npy')
    ytr = np.load('./ytr.npy')
    yte = np.load('./yte.npy')

    return xtr, xte, ytr, yte
