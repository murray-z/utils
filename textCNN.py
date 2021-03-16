# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data_helper_textCNN import *
from sklearn.metrics import classification_report


class Config(object):

    """配置参数"""
    def __init__(self, use_clean_data=False, use_pre_embedding=False):
        self.embedding_pretrained = None

        if use_clean_data:
            self.word2idx = load_json(word2idx_path_clean)
            self.label2idx = load_json(label2idx_path_clean)
            self.model_path_best_loss = "./model_cnn/model_best_loss_clean.pth"
            self.model_path_best_recall = "./model_cnn/model_best_recall_clean.pth"
            if use_pre_embedding:
                self.embedding_pretrained = torch.load(embedding_path_clean)
        else:
            self.word2idx = load_json(word2idx_path)
            self.label2idx = load_json(label2idx_path)
            self.model_path_best_loss = "./model_cnn/model_best_loss.pth"
            self.model_path_best_recall = "./model_cnn/model_best_recall.pth"
            if use_pre_embedding:
                self.embedding_pretrained = torch.load(embedding_path)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(self.label2idx)                          # 类别数
        self.n_vocab = len(self.word2idx)                               # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 100                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = 300                                           # embed_size
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 128                                          # 卷积核数量(channels数)



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def cal_precision_recall(true_label, pred_label):
    # 事件文本作为一个整体计算召回率，准确率
    # print("\n\n事件文本作为一个整体，非事件作为一个整体，二分类：")
    event = [0., 0.]
    none = [0., 0.]
    for t, p in zip(true_label, pred_label):
        if p != "__label__other":
            if t != "__label__other":
                event[0] += 1
            else:
                event[1] += 1
        else:
            if t != "__label__other":
                none[0] += 1
            else:
                none[1] += 1
    try:
        precision = event[0] / (event[0] + event[1])
        recall = event[0] / (event[0] + none[0])
    except:
        precision, recall = 0., 0.
    return precision, recall


def dev(data_loader, model, config):
    model.eval()
    loss_total = 0.
    acc_total = 0.
    batch_num = 0.
    sample_num = 0.
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for texts, labels in data_loader:
            logits = model(texts.to(config.device))
            loss = F.cross_entropy(logits, labels.to(config.device))
            loss_total += loss.item()

            labels = labels.data.cpu().numpy()
            true_labels.extend(list(labels))

            preds = torch.argmax(logits, dim=1)
            preds = preds.data.cpu().numpy()
            pred_labels.extend(list(preds))

            acc = np.sum(preds == labels)
            acc_total += acc
            sample_num += len(labels)
            batch_num += 1

    idx2label = {idx: label for label, idx in config.label2idx.items()}
    true_labels = [idx2label[idx] for idx in true_labels]
    pred_labels = [idx2label[idx] for idx in pred_labels]

    # 二分类
    precision, recall = cal_precision_recall(true_labels, pred_labels)

    return acc_total/sample_num, loss_total/batch_num,  precision, recall, true_labels, pred_labels


def test(config, model_path, data_loader):
    model = Model(config)
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)
    acc, loss, precision, recall, true_labels, pred_labels = dev(data_loader, model, config)
    table = classification_report(true_labels, pred_labels)
    print(table)
    print("\nTEST: acc: {} loss: {} binary: precision: {} recall: {}".format(acc, loss, precision, recall))


def train(config, train_data_path, test_data_path, dev_data_path):
    model = Model(config)

    train_loader = DataLoader(CnnDataSet(train_data_path, config.word2idx, config.label2idx),
                              batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(CnnDataSet(dev_data_path, config.word2idx, config.label2idx),
                            batch_size=config.batch_size, shuffle=False)

    model.train()
    model.to(config.device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    dev_best_loss = float('inf')
    dev_best_recall = float('-inf')

    for epoch in range(config.num_epochs):
        print("\n\nEPOCH:{}".format(epoch))
        for i, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            logits = model(texts)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                labels = labels.data.cpu().numpy()
                preds = torch.argmax(logits, dim=1)
                preds = preds.data.cpu().numpy()
                acc = np.sum(preds == labels)*1./len(preds)
                print("train: epoch: {} step: {} acc: {} loss: {} ".format(epoch+1, i, acc, loss.item()))

                dev_acc, dev_loss, dev_binary_precision, dev_binary_recall, _, _ = dev(dev_loader, model, config)

                print("DEV: acc: {} loss: {}  binary: precision: {} recall: {}".format(
                    dev_acc, dev_loss, dev_binary_precision, dev_binary_recall))

                if dev_loss < dev_best_loss:
                    print("epoch: {} step: {} best loss save model .....".format(epoch, i))
                    torch.save(model.state_dict(), config.model_path_best_loss)
                    dev_best_loss = dev_loss

                # 随机初始化，会导致最开始recall最大
                if epoch == 0 and i < 200:
                    continue
                if dev_binary_recall > dev_best_recall:
                    print("epoch: {} step: {} best recall save model .....".format(epoch, i))
                    torch.save(model.state_dict(), config.model_path_best_recall)
                    dev_best_recall = dev_binary_recall

    test_loader = DataLoader(CnnDataSet(test_data_path, config.word2idx, config.label2idx),
                             batch_size=config.batch_size, shuffle=False)
    print("\n\n测试dev_best_loss生成模型:\n")
    test(config, config.model_path_best_loss, test_loader)

    test_loader = DataLoader(CnnDataSet(test_data_path, config.word2idx, config.label2idx),
                             batch_size=config.batch_size, shuffle=False)
    print("\n\n测试dev_best_recall生成模型:\n")
    test(config, config.model_path_best_recall, test_loader)


if __name__ == "__main__":
    print("\n\n无清洗文本:\n")
    config = Config(use_pre_embedding=False, use_clean_data=False)
    train(config, train_data_path, test_data_path, dev_data_path)

    print("\n\n清洗后文本：\n")
    config = Config(use_pre_embedding=False, use_clean_data=True)
    train(config, train_data_path_clean, test_data_path_clean, dev_data_path_clean)





