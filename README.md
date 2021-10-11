# 华中科技大学 19 级 机器学习相关作业&课设

> 据观察每年的作业可能都不一样, 请确认我们的题目是否相同. 课堂为 机器学习 2021

说明: `educoder-tasks` 文件夹下:

- `*.detail.py` 文件是带注释和较为详细说明和测试数据的文件, **不可**直接复制过去跑
- `*.ans.py` 文件为太长不看版, 可以直接复制, 提交即可通过
- 如果只有 py 文件, 说明这题太简单了没啥好说的.....

**<big>hxd 顺手点个 star 呗 ∠( ᐛ 」∠)\_</big>**

## 目录结构

```txt
|
+--+ educoder-tasks       # educoder 平台上的作业
   |
   +--+ 1-机器学习 --- kNN算法
   |  |
   |  +--- t-1.ans.py         # 第1关：实现kNN算法
   |  +--- t-1.detail.py
   |  +--- t-2.ans.py         # 第2关：红酒分类
   |  +--- t-2.detail.py
   |
   +--+ 2-机器学习之kNN算法
   |  |
   |  +--- t-1.md             # kNN算法原理
   |  +--- t-2.py             # 使用 sklearn 中的kNN算法进行分类
   |  +--- t-3.py             # 使用 sklearn 中的kNN算法进行回归
   |  +--- t-4.py             # 分析红酒数据
   |  +--- t-5.py             # 对数据进行标准化
   |  #    t-6.py (没写)      # 使用kNN算法进行预测, 答案同 `../1-机器学习 --- kNN算法/t-2.ans.py`
   |
   +--+ 3-机器学习 --- 感知机
      |
      +--- t-1.ans.py             # 感知机 - 西瓜好坏自动识别
      +--- t-1.detail.py
```

## 万字长文(详细到像素), 手把手教你配置环境

> 如果你只想要答案交作业而不用本地调试可跳过

1. 下载并安装 python3.x
2. pip 下载 `numpy`, `sklearn`
