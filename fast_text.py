# -*- coding: utf-8 -*-

import fasttext
from sklearn.metrics import classification_report
from data_helper_fasttext import *


"""
input :训练文件路径
lr :学习率 默认为0.1 √
dim : 词向量维度 默认为100
ws :（windows size） 窗口大小 默认为5
minCount : 最小词数 默认为1√
minCountLabel :=1 √ 最小标签数
minn :=0√
maxn :=0√
neg :负例采样个数 默认为5
wordNgrams :最大的词n-gram长度 默认为1
loss :损失函数 {ns,hs,softmax,ova} 默认是softmax√
bucket : 桶数默认为2000000
thread：cpu线程数 默认为12
lrUpdateRate：学习率更新，默认为100
t：负采样阈值 默认0.0001
label :标签前缀 默认 '_ label _'√
verbose：=2 控制打印输出 2显示每个epoch 1显示最后一个epoch
pretrainedVectors:用于监督学习的预训练词向量（.vec文件），给出路径 ，默认为‘ ’
"""


def train(lr, dim, epoch, wordNgrams=3, minCount=1, thread=30, pretrain_vector="", train_data_path="", dev_data_path=""):
    model = fasttext.train_supervised(input=train_data_path,
                                   lr=lr,
                                   dim=dim,
                                   epoch=epoch,
                                   wordNgrams=wordNgrams,
                                   label="__label__",
                                   minCount=minCount,
                                   thread=thread,
                                   pretrainedVectors=pretrain_vector,
                                   verbose=2
                                   )

    labels, texts = get_label_texts(dev_data_path)
    pred_label, pred_prob = model.predict(texts)
    precision, recall = cal_precision_recall(labels, pred_label)

    return precision, recall, model


def cal_precision_recall(true_label, pred_label):
    # 事件文本作为一个整体计算召回率，准确率
    # print("\n\n事件文本作为一个整体，非事件作为一个整体，二分类：")
    event = [0., 0.]
    none = [0., 0.]
    for t, p in zip(true_label, pred_label):
        if p[0] != "__label__other":
            if t != "__label__other":
                event[0] += 1
            else:
                event[1] += 1
        else:
            if t != "__label__other":
                none[0] += 1
            else:
                none[1] += 1
    precision = event[0] / (event[0] + event[1])
    recall = event[0] / (event[0] + none[0])
    return precision, recall


def Test(data_path, model_path):
    labels, texts = get_label_texts(data_path)
    model = fasttext.load_model(model_path)
    pred_label, pred_prob = model.predict(texts)
    precision, recall = cal_precision_recall(labels, pred_label)
    table = classification_report(labels, [p[0] for p in pred_label])
    print(table)
    print("TEST: precision: {}  recall: {}".format(precision, recall))


if __name__ == '__main__':
    lr = 0.5
    dim = 128
    wordNgrams = 5
    minCount = 1
    model_path = "./model_fasttext/model.bin"
    model_path_clean = "./model_fasttext/model_clean.bin"

    # clean数据集
    best_recall = 0.
    for epoch in range(5, 55, 5):
        print("epoch: {}".format(epoch))
        precision, recall, model = train(lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams,
                                         train_data_path=train_data_path_clean, dev_data_path=dev_data_path_clean)
        print("DEV: precision: {}  recall: {}".format(precision, recall))
        if recall > best_recall:
            model.save_model(model_path_clean)
            best_recall = recall
    Test(test_data_path_clean, model_path_clean)

    print("========="*5)

    # normal 数据集
    best_recall = 0.
    for epoch in range(5, 55, 5):
        print("epoch: {}".format(epoch))
        precision, recall, model = train(lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams,
                                         train_data_path=train_data_path, dev_data_path=dev_data_path)
        print("DEV: precision: {}  recall: {}".format(precision, recall))
        if recall > best_recall:
            model.save_model(model_path)
            best_recall = recall
    Test(test_data_path, model_path)


