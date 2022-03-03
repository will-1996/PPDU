import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import metrics
import random
import copy

Size = 4039


def Random_R(graph, epsilon):
    """
    随机响应扰动方法
    :param graph: 只能为二元矩阵，0代表无边，1代表有边
    :param epsilon:
    :return:
    """
    p = np.exp(epsilon)/(1+np.exp(epsilon))
    inverse_matrix = (np.random.rand(len(graph), len(graph[0])) > p).astype(int)
    print(inverse_matrix)
    graph = np.abs(graph-inverse_matrix)
    return graph



graph = [[0, 1, 1, 0, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0]]

a = Random_R(graph=graph, epsilon=1)
for i in range(len(graph)):
    a[i][i]=0
print(a)