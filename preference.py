"""
------------No one knows my privacy better than myself----------------
"""
import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import metrics
import random


Size = 4039
# Ep = 1


def Random_R(graph, epsilon):
    """
    随机响应扰动方法
    :param graph: 只能为二元矩阵，0代表无边，1代表有边
    :param epsilon:
    :return:
    """
    p = np.exp(epsilon)/(1+np.exp(epsilon))
    inverse_matrix = (np.random.rand(len(graph), len(graph[0])) > p).astype(int)
    graph = np.abs(graph-inverse_matrix)
    return graph


def R_R(a, epsilon):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    if np.random.rand() > p:
        a = 1-a
    return a


def get_seq(read_path):
    f = open(read_path, 'r')
    seq = []
    for line in f:
        a = line.strip().split(',')
        a = [int(i) for i in a]
        # map(int, a)
        seq.append(a)
    return seq


def label_matr_gen(node_num, clu_num, res):
    label_matrix = np.zeros([node_num, clu_num])
    for i in range(len(res)):
        for j in res[i]:
            label_matrix[int(j)][i] = 1
    return label_matrix


def Modularity(S, A):
    """
    :param S: label matrix
    :param A: Adjacency matrix, 非三角矩阵
    :return:
    """
    m = np.sum(A)/2         # 总边数
    k = np.sum(A, axis=1)   # 度
    B = A - np.outer(k, k)/(2*m)
    Q = np.trace(np.dot(np.dot(np.transpose(S), B), S))/(2*m)
    return Q


def label_gen(re, n):
    label = np.zeros(n)
    for c in range(len(re)):
        for i in re[c]:
            label[i] = c
    return label


def graphPert(graph, epsilon):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    # p=0.001
    inte = Size//2+1
    for i in range(Size):
        if i <= Size - inte:  #  ，左右都需置0
            for j in range(i + inte, Size):
                graph[i, j] = 0
        # 无论i处在哪里， i左边都不属于其负责。
        for j in range((i + inte) % Size, i):
            graph[i, j] = 0
    for i in range(Size):
        for j in range(Size):
            if graph[i, j] != 0:
                if np.random.rand() > p:
                    graph[i, j] = 0
    return graph


def rabv(graph, epsilon):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    inte = Size // 2 + 1
    # 将user i不负责的edge置为0
    for i in range(Size):
        if i <= Size - inte:  # i负责的edge处在整行的中间，左右都需置0
            for j in range(i + inte, Size):
                graph[i, j] = 0
        # 无论i处在哪里， i左边都不属于其负责。
        for j in range((i + inte) % Size, i):
            graph[i, j] = 0

    # 对user i负责的所有edge进行扰动
    for i in range(Size):
        if i <= Size - inte:        # i负责的edge在整行中间
            for j in range(i, i+inte):
                if np.random.rand() > p:
                    graph[i, j] = 1-graph[i, j]
        else:                       # i负责的edge分散在行的两端，需要分别扰动
            for j in range((i+inte)%Size):
                if np.random.rand() > p:
                    graph[i, j] = 1-graph[i, j]
            for j in range(i+inte, Size):
                if np.random.rand() > p:
                    graph[i, j] = 1-graph[i, j]
    return graph


def graphPertPro(graph, epsilon, fake, degree):
    """
    不止扰动已经存在的边，还从可能存在的边中，sample出来部分进行保护
    :param graph:
    :param epsilon:
    :param fake: 伪边相对于真实存在边的比例
    :return:
    """
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    # 每个用户负责n/2个边信息的扰动
    inte = Size//2+1
    for i in range(Size):
        if i <= Size - inte:      # i负责的edge处在整行的中间，左右都需置0
            for j in range(i+inte, Size):
                graph[i, j] = 0
        # 无论i处在哪里， i的直接左边都不属于其负责。
        for j in range((i+inte) % Size, i):
            graph[i, j] = 0
    # 先完成对已经存在边的扰动
    for i in range(Size):
        for j in range(Size):
            if graph[i, j] != 0:
                if np.random.rand() > p:
                    graph[i, j] = 0

    # 从负责范围内随机选取定量的边近似作为fake边，进行扰动
    for i in range(Size):
        if i <= Size - inte:        # i负责的edge在行中间，从这中间采样fake边进行扰动
            if degree[i]*(fake+1)<inte:     # 需要扰动的边数量小于负责边总数时：采样
                ind = random.sample(range(i, i + inte), fake * degree[i])
                for j in ind:
                    if np.random.rand() > p:
                        graph[i, j] = 1-graph[i, j]
            else:                           # 需要扰动的边数大于负责总边数：对所有负责边进行扰动
                for j in range(i, i+inte):
                    if np.random.rand() > p:
                        graph[i, j] = 1 - graph[i, j]
        else:                               # i负责的edge在两端，利用求余数完成对负责边的索引
            if degree[i] * (fake + 1) < inte:  # 需要扰动的边数量小于负责边总数时：采样
                ind = np.array(ind) % Size
                for j in ind:
                    if np.random.rand() > p:
                        graph[i, j] = 1 - graph[i, j]
            else:                             # 扰动边数大于总负责边数，全扰动
                for j in range(i):
                    if np.random.rand() > p:
                        graph[i, j] = 1 - graph[i, j]
                for j in range(i+inte, Size):
                    if np.random.rand() > p:
                        graph[i, j] = 1 - graph[i, j]
    return graph


def graphRecover(graph):
    for i in range(Size):
        for j in range(Size):
            if graph[i, j] == 1:
                graph[j, i] = 1
    return graph


def graphToSeq(graph):
    seq = []
    for i in range(len(graph)):
        seq.append([])
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i, j] == 1:
                seq[i].append(j)
    return seq


def nxGraphGen(nodes, seq):
    edges = []
    for i in range(Size):
        # edges.append([])
        for j in seq[i]:
            edges.append([i, j])
    G = nx.Graph()
    # 往图添加节点和边
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def CC(G):
    cc = nx.clustering(G)
    # print(cc.values())
    total = 0
    for i in range(Size):
        total += cc[i]
    return total/Size


def louvainClustering(seq, nodes):
    G = nxGraphGen(nodes, seq)
    # cc = CC(G)
    partition = community_louvain.best_partition(G)
    resu = []
    for i in range(Size):
        resu.append([])
    for i in range(Size):
        resu[partition[i]].append(i)
    resu = [i for i in resu if i != []]
    return resu


def label_gen(re, n):
    label = np.zeros(n)
    for c in range(len(re)):
        for i in re[c]:
            label[i] = c
    return label


Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
nodes = np.arange(Size)
Seq = get_seq('data/facebook_seq')
Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)


# ------------ground truth计算-------------------
ground_resu = louvainClustering(Seq, nodes)
print(ground_resu)
ground_matrix = label_matr_gen(Size, len(ground_resu), ground_resu)
ground_label = label_gen(ground_resu, Size)

lis_Q = []
lis_R = []
lis_M = []
for Ep in range(1, 9):
    # Ep = Ep/10
    # ---------------graph扰动及Sequence计算-------------------
    # graph1 = rabv(graph=Graph, epsilon=Ep)                 # RABV-only, baseline方法
    graph1 = graphPert(graph=Graph, epsilon=Ep)         # 不考虑不存在的边
    graph1 = graphPertPro(graph=Graph, epsilon=Ep, fake=4, degree=Degree)
    graph1 = graphRecover(graph1)
    seq1 = graphToSeq(graph1)

    # ---------------------louvain聚类----------------------------
    resu = louvainClustering(seq1, nodes)

    # -----------------------模块度计算----------------------------
    label_matrix = label_matr_gen(Size, len(resu), resu)
    Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
    Q = Modularity(label_matrix, Graph)
    print('Modularity is:', Q)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu, Size)
    ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)

    lis_Q.append(Q)
    lis_R.append(ari_score)
    lis_M.append(ami_score)

print(lis_Q)
print(lis_R)
print(lis_M)







