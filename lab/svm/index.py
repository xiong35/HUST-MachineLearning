from sklearn.svm import SVC
import jieba as jb
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
vector_dimension = 300


### main ###
print("main")

#### load data ####
print("load data")
##### train data #####
print("train data")
df = pd.read_csv("../data/train.csv")
df.fillna('', inplace=True)

train_data = []
train_label = []

for i in range(0, df.shape[0]):
    train_data.append(jb.lcut(df["content"][i] + df["comment_all"][i]))
    train_label.append(df["label"][i])

##### test data #####
print("test data")
test_data = []
df = pd.read_csv("../data/test.csv")
df.fillna('', inplace=True)

for i in range(0, df.shape[0]):
    test_data.append(jb.lcut(df["content"][i] + df["comment_all"][i]))

#### word to vector ####
print("word to vector")
x = []

total = []
total.extend(train_data)
total.extend(test_data)

for i, data in enumerate(total):
    x.append(TaggedDocument(data, ['Text' + '_%s' % str(i)]))

text_model = Doc2Vec(vector_size=vector_dimension)
text_model.build_vocab(x)
text_model.train(x, total_examples=text_model.corpus_count,
                 epochs=10)

train_size = len(train_data)
test_size = len(test_data)

train_arrays = np.zeros((train_size, vector_dimension))
test_arrays = np.zeros((test_size, vector_dimension))
train_labels = np.zeros(train_size)
test_labels = np.zeros(test_size)

for i in range(train_size):
    train_arrays[i] = text_model.docvecs['Text_' + str(i)]
    train_labels[i] = train_label[i]

j = 0
for i in range(train_size, train_size + test_size):
    test_arrays[j] = text_model.docvecs['Text_' + str(i)]
    j = j + 1


#### fit model ####
print("fit model")
clf = SVC()
clf.fit(train_arrays, train_labels)
y_pred = clf.predict(test_arrays)
np.savetxt('./result.txt', y_pred, fmt="%d", delimiter=" ")
