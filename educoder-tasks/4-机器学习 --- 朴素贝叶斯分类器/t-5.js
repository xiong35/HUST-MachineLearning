var code = `
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def news_predict(train_sample, train_label, test_sample):
    vec = CountVectorizer()
    X_train_count_vectorizer = vec.fit_transform(train_sample)
    X_test_count_vectorizer = vec.transform(test_sample)

    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train_count_vectorizer)
    X_test = vec.transform(X_test_count_vectorizer)

    clf = MultinomialNB()

    clf.fit(X_train, train_label)
    result = clf.predict(X_test)

`
  .replaceAll("\n", "\\n")
  .replaceAll('"', '\\"');

fetch("https://data.educoder.net/api/myshixuns/yv9anftf4r/update_file.json", {
  headers: {
    accept: "application/json",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "cache-control": "no-cache",
    "content-type": "application/json; charset=utf-8",
    pragma: "no-cache",
    prefer: "safe",
    "sec-ch-ua":
      '"Microsoft Edge";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
  },
  referrer: "https://www.educoder.net/tasks/r2a6zphucbm3",
  referrerPolicy: "unsafe-url",
  body: `{"path":"step5/student.py","evaluate":0,"content":"${code}","game_id":24420487}`,
  method: "POST",
  mode: "cors",
  credentials: "include",
});
