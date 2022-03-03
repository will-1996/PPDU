import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import metrics
import random
import copy

Size = 4039

Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)


# Ind = Degree.argsort()
# # print(Degree.argsort())
# for i in range(Size):
#     if Degree[Ind[i]] > 1:
#         start = i
#         break
# degree1_ind = Ind[:start]
# Ind = Ind[start:]               # 存储的是度值大于等于2的结点的索引
#
# # Ind = Ind[np.argsort(Degree)]
# def phasePre(degree, ind):
#     re = []
#     temp = 0
#     length = degree[ind[temp]] + 1
#     while temp + length < len(ind):
#         # length = degree[temp]+1
#         end = temp + length
#         re.append(ind[temp:end])
#         temp = end
#         length = degree[ind[temp]] + 1
#     re.append(ind[temp:])
#     return re
#
#

# re = phasePre(Degree, Ind)
#
# # print(phasePre(Degree, Ind))
# for i in range(len(re)):
#     for j in range(len(re[i])):
#         pass

# --------------一阶零模型------------------------------
def DGG_Gen(Degree):
    """
    利用度序列和一阶零模型生成合成图
    :param Degree:
    :return:
    """
    Graph_empty = np.zeros([Size, Size])
    deg = copy.deepcopy(Degree)
    # m = np.sum(deg)//2
    for i in range(Size):
        total = np.sum(deg[i+1:])
        if not total:
            break
        remain_nodes = list(range(i+1, Size))
        porb = deg[i+1:]/np.sum(deg[i+1:])
        # print(porb)
        connected_edges = np.random.choice(remain_nodes, size=(deg[i]), replace=False, p=porb)
        # a = np.random.randint(low=0, high=total, size=)
        for j in connected_edges:
            deg[j] -= 1
            Graph_empty[i, j] = 1
    for i in range(Size):
        for j in range(i+1, Size):
            Graph_empty[j, i] = Graph_empty[i, j]
    return Graph_empty


# print(np.sum(Graph_empty, axis=1))
# print(Degree)
    




# b = [10, 20, 1]
# bb = [1, 2, 3]
# c = b/np.sum(b)
# a = np.random.choice(bb, size=(2), replace=False, p=c)
# # print(a)
# 
# x = list(range(3, 10))
# print(x)