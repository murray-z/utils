# -*- coding: utf-8 -*-

""""对高维数据降维到二维平面，以散点图进行显示"""


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_scatter(data, label, show_num=50, title="scatter"):
    """
    :param data: 显示数据； 格式： [[1,2,3], [2,2,4], [4,5,6]]
    :param label: 数据标签；格式： ['体育', '娱乐', '财经']
    :param show_num: 每个类别显示数目
    :param title: 图片标题
    :return:
    """

    # 颜色
    color = ['b', 'r', 'c', 'm', 'g', 'b', 'r', 'c', 'm', 'g']
    # 形状
    marker = ['o', '+', '.', 's', 'x', '1', 'p', '*', '>', 'd']

    # 数据降维
    pca = PCA(n_components=2)
    data = pca.fit_transform(data).tolist()

    # 获得每个类别要显示个数的数据
    counter = {}
    categary_data = {}
    for label, data in zip(label, data):
        if label not in counter:
            counter[label] = 1
            categary_data[label] = [data]
        else:
            if counter[label] == show_num:
                break
            else:
                counter[label] += 1
                categary_data[label].append(data)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title(title)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    idx = 0
    for label, data in categary_data.items():
        x = [item[0] for item in data]
        y = [item[1] for item in data]
        ax1.scatter(x, y, c=color[idx], marker=marker[idx], label=label)
        idx+=1
    # 设置图标
    plt.legend(loc='best')
    # 显示所画的图
    plt.show()


if __name__ == '__main__':
    plot_scatter([[1,2,3], [2,2,4], [4,5,6]], ['体育', '娱乐', '财经'])