var code = `import numpy as np


class NaiveBayesClassifier(object):

    def __init__(self):

        self.label_prob = {}

        self.condition_prob = {}

    def fit(self, feature, label):
        for l in label:
            self.label_prob[l] = self.label_prob.get(l, 0) + 1

        for k, v in self.label_prob.items():
            self.label_prob[k] = (v+1) / (len(label) +
                                          len(self.label_prob.keys()))

        label2data = {}
        for i, data in enumerate(feature):
            l = label[i]
            old_data = label2data.get(l)
            if not old_data:
                label2data[l] = [data]
            else:
                label2data[l].append(data)

        for l, all_data in label2data.items():
            feat_index2feat_count = {}
            for i in range(len(feature[0])):
                feat_index2feat_count[i] = {}
                for f in feature:
                    d = f[i]
                    feat_index2feat_count[i][d] = 0

            for data in all_data:
                for i, d in enumerate(data):
                    feat_index2feat_count[i][d] = feat_index2feat_count[i].get(
                        d, 0) + 1

            for i in feat_index2feat_count.keys():
                for k in feat_index2feat_count[i].keys():
                    feat_index2feat_count[i][k] = (
                        feat_index2feat_count[i][k]+1) / (len(all_data)+len(feat_index2feat_count[i].keys()))

            self.condition_prob[l] = feat_index2feat_count

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        ret_arr = []

        for fs in feature:
            label2res = {}
            for label, feat_index2feat_count in self.condition_prob.items():
                p = self.label_prob[label]

                for i, f in enumerate(fs):
                    p *= (feat_index2feat_count[i].get(f, 0))

                label2res[label] = p


            max_prob = {"label": "foo", "prob": -1}
            for label, prob in label2res.items():
                if(prob > max_prob["prob"]):
                    max_prob = {"label": label, "prob": prob}

            ret_arr.append(max_prob["label"])

        return ret_arr


feat = [[2, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 1, 2],
        [1, 2, 3]]

label = [1, 0, 1, 0, 1]

test = [[2, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 100, 2],
        [1, 2, 3]]


p = NaiveBayesClassifier()
p.fit(feat, label)
res = p.predict(test)



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
  referrer: "https://www.educoder.net/tasks/43ac57wnjh9o",
  referrerPolicy: "unsafe-url",
  body: `{"path":"step4/student.py","evaluate":0,"content":"${code}","game_id":24420486}`,
  method: "POST",
  mode: "cors",
  credentials: "include",
});
