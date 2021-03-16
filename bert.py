# -*- coding: utf-8 -*-


import torch.nn as nn
from transformers import BertModel, AdamW
from data_helper_bert import *
from sklearn.metrics import classification_report

class Config():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label2idx = load_json(label2idx_path)
        self.class_num = len(self.label2idx)
        self.max_seq_len = 100
        self.epochs = 5
        self.lr = 2e-5
        self.hidden_size = 768
        self.dropout = 0.1
        self.batch_size = 32
        self.best_acc_model_path = "./model_bert/best_acc.pth"
        self.best_recall_model_path = "./model_bert/best_recall.pth"


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Linear(config.hidden_size, config.class_num)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        out = self.fc(pooled_output)
        return out


def cal_acc(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    preds = torch.argmax(preds, dim=1)
    # print(preds, labels)
    return torch.sum(preds == labels)*1. / len(labels)


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



def dev(model, data_loader, config, test=False):
    device = config.device
    idx2label = {idx: label for label, idx in config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[
                2].to(device), batch[3].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    # print(idx2label, pred_labels, true_labels)
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]

    acc = sum([1 if p==t else 0 for p, t in zip(pred_labels, true_labels)]) *1. / len(pred_labels)

    binary_precison, binary_recall = cal_precision_recall(true_labels, pred_labels)

    if test:
        return pred_labels, true_labels, acc, binary_precison, binary_recall
    else:
        return acc, binary_precison, binary_recall


def Test(model_path, config):
    test_dataset = BertDataSet(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model = Bert(config)
    model.load_state_dict(torch.load(model_path))

    pred_labels, true_labels, acc, binary_precison, binary_recall = dev(model, test_dataloader, config, test=True)

    table = classification_report(true_labels, pred_labels)
    print(table)
    print("TEST: acc: {}  binary_precision: {}  binary_recall: {}".format(acc, binary_precison, binary_recall))



def train():
    config = Config()

    model = Bert(config)

    device = config.device

    train_dataset = BertDataSet(train_data_path)
    dev_dataset = BertDataSet(dev_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    best_acc = 0.
    best_recall = 0.

    for epoch in range(config.epochs):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(
                device), batch[3].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                acc = cal_acc(logits, labels)
                print("TRAIN: epoch: {} step: {} acc: {}, loss: {}".format(epoch, i, acc, loss.item()))

        acc, binary_precision, binary_recall = dev(model, dev_dataloader, config)
        print("DEV: epoch: {} acc: {} binary_precision: {} binary_recall: {}".format(epoch, acc,
                                                                                     binary_precision, binary_recall))
        if acc > best_acc:
            torch.save(model.state_dict(), config.best_acc_model_path)
            best_acc = acc

        if binary_recall > best_recall:
            torch.save(model.state_dict(), config.best_recall_model_path)
            best_recall = binary_recall

    # 根据最优acc测试
    Test(config.best_acc_model_path, config)
    # 根据最优recall测试
    Test(config.best_recall_model_path, config)


if __name__ == "__main__":
    train()

