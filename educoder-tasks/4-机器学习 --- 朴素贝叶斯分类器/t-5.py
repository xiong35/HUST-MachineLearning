
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def news_predict(train_sample, train_label, test_sample):
    vec = CountVectorizer()
    X_train_count_vectorizer = vec.fit_transform(train_sample)
    X_test_count_vectorizer = vec.transform(test_sample)

    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train_count_vectorizer)
    X_test = tfidf.transform(X_test_count_vectorizer)

    clf = MultinomialNB(0.03)

    clf.fit(X_train, train_label)
    return clf.predict(X_test)
