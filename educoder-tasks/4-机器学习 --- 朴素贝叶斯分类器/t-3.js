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
  referrer: "https://www.educoder.net/tasks/fa3x6tpnwjor",
  referrerPolicy: "unsafe-url",
  body:
    '{"path":"step3/student.py","evaluate":0,"content":"import numpy as np\\n\\n\\nclass NaiveBayesClassifier(object):\\n    \\n    def __init__(self):\\n        self.label_prob = {}\\n        self.condition_prob = {}\\n\\n    def fit(self, feature, label):\\n        for l in label:\\n            self.label_prob[l] = self.label_prob.get(l, 0) + 1\\n        \\n        for k, v in self.label_prob.items():\\n            self.label_prob[k] = v / len(label)\\n\\n        label2data = {}\\n\\n        for i, data in enumerate(feature):\\n            l = label[i]\\n            if not label2data.get(l, None):\\n                label2data[l] = [data]\\n            else:\\n                label2data[l].append(data)\\n        \\n        for l, all_data in label2data.items():\\n            feat_index2feat_count = {}\\n            for data in all_data:\\n                for i, d in enumerate(data):\\n                    feat_index2feat_count[i] = feat_index2feat_count.get(i, {})\\n                    feat_index2feat_count[i][d] = feat_index2feat_count[i].get(d, 0)+1\\n            \\n            feat_index2feat_count[\\"__count__\\"] = len(all_data)\\n\\n            self.condition_prob[l] = feat_index2feat_count\\n                \\n\\n\\n    def predict(self, feature):\\n\\n        ret_arr = []\\n\\n        for fs in feature:\\n            label2res = {}\\n            for label, feat_index2feat_count in self.condition_prob.items():\\n                p = self.label_prob[label]\\n\\n                for i, f in enumerate(fs):\\n                    p *= feat_index2feat_count[i].get(f, 0) / feat_index2feat_count[\\"__count__\\"]\\n\\n                label2res[label] = p\\n\\n            max_prob = {}\\n            for label, prob in label2res.items():\\n                if(prob>max_prob.get(\\"prob\\", -1)):\\n                    max_prob = {\\"label\\": label, \\"prob\\": prob}\\n            ret_arr.append(max_prob[\\"label\\"])\\n        \\n        return ret_arr\\n        # ********* Begin *********#\\n        \\n        #********* End *********#","game_id":24420485}',
  method: "POST",
  mode: "cors",
  credentials: "include",
});
