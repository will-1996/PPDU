import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

data_path = "Aggregation_cluster=7.txt"


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter='\t')
    return points


def cal_dis(data, clu, k):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(m.sqrt((data[i, 0] - clu[j, 0])**2 + (data[i, 1]-clu[j, 1])**2))
    return np.asarray(dis)


def divide(data, dis):
    """
    对数据点分组
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[0]

    return np.asarray(clusterRes)


# def center(data, clusterRes, k):
#     """
#     计算质心
#     :param group: 分组后样本
#     :param k: 类别个数
#     :return: 计算得到的质心
#     """
#     clunew = []
#     for i in range(k):
#         # 计算每个组的新质心
#         idx = np.where(clusterRes == i)
#         sum = data[idx].sum(axis=0)
#         avg_sum = sum/len(data[idx])
#         clunew.append(avg_sum)
#     clunew = np.asarray(clunew)
#     return clunew


def center(data, clusterRes, k, epsilon):
    """
    计算质心
    :param group: 分组后样本
    :param k: 类别个数
    :return: 计算得到的质心
    """
    clunew = []
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterRes == i)
        sum = data[idx].sum(axis=0)
        # 给定 全局敏感度
        # delta = np.array([2, 3, 3, 1, 2, 4, 4, 3, 3, 4])
        delta = np.array([35, 30])
        # 对数据点总和加噪
        noisy_sum = sum + np.random.laplace(loc=0, scale=delta/epsilon, size=(len(delta)))
        noisy_num = np.round(len(data[idx])+np.random.laplace(loc=0, scale=1/epsilon)).astype(int)
        avg_sum = noisy_sum/noisy_num
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew


def classfy(data, clu, k, epsilon):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k, epsilon)
    err = clunew - clu
    return err, clunew, k, clusterRes


def plotRes(data, clusterRes, clusterNum):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param clusterNum: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
    plt.show()


def plot_2(data, clusterRes1, clusterRes2, clusterNum):
    """
    同时绘制两个聚类方法的结果
    :param data:  两种方法的共同数据集
    :param clusterRes1: 方法1的结果
    :param clusterRes2: 方法2的结果
    :param clusterNum: kmeans中的k
    :return: 无返回值
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for j in range(nPoints):
            if clusterRes1[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
            if clusterRes2[j] == i:
                x2.append(data[j, 0])
                y2.append(data[j, 1])
        plt.subplot(2, 1, 1)
        plt.scatter(x1, y1, c=color, alpha=1, s=10, marker='p')
        plt.subplot(2, 1, 2)
        plt.scatter(x2, y2, c=color, alpha=1, s=10, marker='x')
    plt.show()

if __name__ == '__main__':
    k = 7                                          # 类别个数
    data = load_data()
    data = np.array(data)[:, :2]
    
    # print(np.max(data[:, 0]))
    # print(np.max(data[:, 1]))

    # filename = 'data/traveldata.csv'
    # data = []
    # csv_file = csv.reader(open(filename))
    # for row in csv_file:
    #     data.append(row[1:])
    # data = np.array(data[1:])
    # data = data.astype(float)

    total_iteration = 10  # 给定迭代轮次\
    epsilon = 80                              # 给定隐私预算
    clu = random.sample(data[:, :].tolist(), k)  # 随机取质心
    clu = np.asarray(clu)
    err, clunew,  k, clusterRes = classfy(data, clu, k, epsilon)

    for i in range(total_iteration-1):
        if np.any(abs(err) > 0):
            # epsilon = epsilon/2
            err, clunew, k, clusterRes = classfy(data, clunew, k, epsilon/8)
        else:
            break

    print(i)
    # while np.any(abs(err) > 0.1):
    #     print(clunew)
    #     err, clunew, k, clusterRes = classfy(data, clunew, k)

    clulist = cal_dis(data, clunew, k)
    clusterResult = divide(data, clulist)

    # nmi, acc, purity = eva.eva(clusterResult, np.asarray(data[:, 2]))
    # print(nmi, acc, purity)
    pred = KMeans(n_clusters=k, random_state=8).fit_predict(data)

    plot_2(data=data, clusterRes1=clusterResult, clusterRes2=pred, clusterNum=k)
    # plotRes(data, clusterResult, k)
    # plotRes(data, pred, k)