# -*- coding: utf-8 -*-


from sklearn import metrics


def measure(y_test, y_predict, labels):
    """
    :param y_test:  真实的标签
    :param y_predict: 预测标签
    :param labels: 混淆矩阵中显示标签顺序
    :return:
    """
    assert len(y_test) == len(y_predict)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("accuracy: {}".format(accuracy))
    for average in ['micro', 'macro']:
        print('average method: {}'.format(average))
        precision = metrics.precision_score(y_test, y_predict, average=average, labels=labels)
        recall = metrics.recall_score(y_test, y_predict, average=average, labels=labels)
        f1 = metrics.f1_score(y_test, y_predict, average=average, labels=labels)
        print("precision: {} recall: {} f1: {}".format(precision, recall, f1))
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict, labels=labels)

    print("\n")
    print("confusion_matrix:")

    Header = "%5s\t" % ""
    for lab in labels:
        Header += "%5s" % lab + '\t'

    print(Header)

    for idx, lab in enumerate(labels):
        line = "%5s" % lab + '\t'
        for row in list(confusion_matrix[idx]):
            line += '%5d' % row + '\t'
        print(line)

    print("\n\n")


if __name__ == '__main__':
    measure(['a', 'a', 'b', 'a'], ['a', 'a', 'b', 'b'], ['a', 'b'])