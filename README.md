# Semantic-Similarity-Match
此开源hub是基于Tensorflow2.x实现文本相似度匹配等项目

## 1、项目介绍
本项目源于QA对话系统中的文本相似度检索的精排阶段，可抽象为文本相似度匹配任务； 文本相似度匹配中特征的提取一般为静态词向量和动态词向量两种，本项目基于预训练模型的动态词向量；

由于位于检索的精排阶段，考虑到推理时延，需用浅层模型，本项目以Tiny Roberta为 baseline进行实验，后续版本会再次基础上对评价指标进行持续优化，更新中...


## 2、数据集来源

* **数据集来源：[QA_corpus]()**

* **数据集情况**

type     |pair(个)
:-------|---
train |约 10w
valid |约 1w
test |约 1w


## 3、支持模型

*[支持模型](https://github.com/ymcui/Chinese-BERT-wwm):

* **chinese_L-12_H-768_A-12**

* **chinese_rbt4_L-4_H-768_A-12**

* **chinese_rbt6_L-6_H-768_A-12**

* **chinese_rbt12_L-12_H-768_A-12** **等...**

*注：根据情况在 config/.yaml、config.py 中配置


## 4、项目结构

```
.
├── LICENSE
├── README.md
├── chinese_rbt4_L-4_H-768_A-12
│   ├── bert_config_rbt4.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   ├── variable_name_mapping.json
│   └── vocab.txt
├── config
│   └── faq_ranking_model.yaml
├── data
│   ├── similarity_label_vocab.txt
│   ├── test
│   │   ├── test.label
│   │   ├── test.seq1.in
│   │   └── test.seq2.in
│   ├── train
│   │   ├── train.label
│   │   ├── train.seq1.in
│   │   └── train.seq2.in
│   └── valid
│       ├── valid.label
│       ├── valid.seq1.in
│       └── valid.seq2.in
├── data_csv
│   ├── similarity_label_vocab.txt
│   ├── test
│   │   └── test.csv
│   ├── train
│   │   └── train.csv
│   └── valid
│       └── valid.csv
├── models.py
├── nn 
│   ├── activations.py
│   ├── callback.py
│   ├── crf.py
│   ├── layer.py
│   ├── loss.py
│   ├── lr_scheduler.py
│   └── metric.py
├── preprocess
│   ├── config.py
│   ├── data.py
│   └── tokenization.py
├── run_tasks.py
├── utils
│   ├── addict.py
│   ├── get_username.py
│   ├── print_metrics.py
│   └── send_email.py
└── write_data_csv_to_seq_file.py
```


## 5、版本更新
Version |Describe
:-------|---
v1.0 |原始Tiny Roberta：baseline
v2.0 |Big Roberta->distill——>Tiny Roberta

## 6、模型结构

* **finetune-Tiny-Roberta**

![finetune-Tiny-Roberta](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/finetune-Tiny-Roberta-picture.png)


* **distilled-Tiny-Roberta (to do...)**

![distilled-Tiny-Roberta](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/distilled-Tiny-Roberta-picture.png)


## 6、评估结果

运行 run_tasks.py 开始训练.

* **原始Tiny-Roberta finetune下效果:**

![效果1](https://github.com/xuyingjie521/Semantic-Similarity-Match/blob/main/images/test_result1.png)


参考对比：[各种模型评价](https://github.com/terrifyzhao/text_matching)
  

## 交流

本项目作为笔者在之前工作中项目背景下的抽象出的NLP任务demo和trick。 源码和数据（实验数据）已经在项目中给出。

如需要更深一步的交流，请发送消息至邮箱 1812316597@qq.com，或者在 Github 上直接issue。