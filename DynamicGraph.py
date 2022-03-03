"""
动态图更新算法
    节点积累新增的边信息，积累到阈值k以后再进行一次统一更新
    新增节点的情况，新增的节点仅更新自己应该负责的一部分信息
    更新k条边时，考虑这些边对图某些属性的影响，寻找可以替代的备选边
        ---------可以根据具体属性来针对性的寻找备选边，使得生成图的效果有所侧重
    --------不考虑节点删除的情况：节点删除意味着用户退出当前graph，一个用户不存在的graph自然不会泄露其任何隐私
    --------暂不考虑删边的情况：删边需要更新全局图，相关用户需要重新提交关于已存在边的信息
    传统LDP--RR扰动方法，impractical，因为新增user时需要所有用户都提交新的边信息

    考虑baseline方法：

"""
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
        # if type(seq[i]) != list:
        #     edges.append([i, seq[i]])
        #     continue
        for j in seq[i]:
            j = int(j)
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


def random_sam(leng, se, num):
    """
    为单个用户指定隐私范围
    :param leng: graph size n
    :param se: IDs of existing edges
    :param num: delta--parameter of privacy scope
    :return:  privacy scope
    """
    seq = list(range(leng))
    if num+len(se) >= leng:
        return seq
    re = list(set(seq)-set(se))
    # 从non-existing edges中采样privacy edges
    re = random.choices(re, k=num)
    # 将existing edges加入privacy edges
    re.extend(se)
    # 对privacy scope排序
    re.sort()
    return re


def priSpe(graph, degree, delta):
    """
    指定所有结点的隐私范围
    :param graph: 原始graph
    :param delta: 隐私倍率
    :return: 隐私范围
    """
    ps = []
    siz = len(graph)
    degree_p = degree*delta
    for i in range(len(graph)):
        se = [j for j in range(siz) if graph[i, j] == 1]
        ps.append(random_sam(leng=siz, se=se, num=degree_p[i]))
    return ps


def Pert(graph, ps, epsilon):
    """

    :param graph: 原始graph
    :param ps:  privacy scope，隐私范围
    :param epsilon: 隐私预算
    :return:
    """
    # 设定扰动概率
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    for i in range(len(graph)):
        for j in ps[i]:         # 遍历隐私范围内的元素，逐个RR扰动
            if np.random.rand() > p:
                graph[i, j] = 1-graph[i, j]
    return graph


# def graphPartition(graph, num, k, degree):
#     """
#     从原始graph中删除部分边
#     :param graph: 原始图
#     :param num: 删边的节点数量
#     :param k: 每个节点删除的边数
#     :param degree: 度序列
#     :return: 删除部分边之后的图-and-删边的节点索引
#     """
#     bignodes = []
#     swaps = []
#     for i in range(Size):
#         if degree[i] >= 3*k:
#             bignodes.append(i)
#     # bignodes = set(bignodes)
#     lucknodes = random.choices(bignodes, k=num)
#     # print(lucknodes)
#     partGraph = copy.deepcopy(graph)
#     # partGraph = np.array(partGraph)
#     # print(type(partGraph))
#     for i in lucknodes:
#         ind = []
#         for j in range(Size):
#             if partGraph[i, j] == 1:
#                ind.append(j)
#         swap = random.choices(ind, k=k)
#         swaps.append(swap)
#         for j in swap:
#             partGraph[i, j] = 1-partGraph[i, j]
#             partGraph[j, i] = partGraph[i, j]
#     return partGraph, lucknodes, swaps


def graphPartition(graph, num, k, degree):
    """
    从原始graph中删除部分边
    :param graph: 原始图
    :param num: 删边的节点数量
    :param k: 每个节点删除的边数
    :param degree: 度序列
    :return: 删除部分边之后的图-and-删边的节点索引
    """
    bignodes = []
    swaps = []
    for i in range(Size):
        if degree[i] >= 3*k:
            bignodes.append(i)

    lucknodes = random.choices(bignodes, k=num)
    partGraph = copy.deepcopy(graph)
    for i in lucknodes:
        ind = []
        for j in range(Size):
            if partGraph[i, j] == 1:
               ind.append(j)            # ind中还应该包含部分属于privacy scope但non-existing 的边-------------

        count = 0                       # 在ind中添加
        for j in range(Size):
            a = np.random.randint(low=0, high=Size)
            if partGraph[i, a] == 0:
                ind.append(a)
                count += 1
            if count >= degree[i]//2:
                break

        a = np.round(k + np.random.laplace(0, 1)).astype(int)                   # ----------attention！！！ 这里的laplace参数简单的置为了1------------
        # ------------保证边改变数量a的区间在 0~3*k之间------------
        if a > 3*k:
            a = 3*k
        if a <=0:
            a = 1
        swap = random.choices(ind, k=a)           # swap：改变状态的边的索引， 此处的ind包含1.5d个元素， 而d>3k, a<3k
        swaps.append(swap)
        for j in swap:
            partGraph[i, j] = 1-partGraph[i, j]
            partGraph[j, i] = partGraph[i, j]
    return partGraph, lucknodes, swaps


def nodeDelte(graph, num):
    """
    从原始graph中删除部分节点
    :param graph:原始图
    :param num: 删除结点的数量
    :return:
    """
    graphDel = copy.deepcopy(graph)
    # graphDel = np.array(graphDel)
    del_index = np.random.randint(low=0, high=Size, size=num)
    for i in del_index:
        for j in range(Size):
            graphDel[i, j] = 0
    return graphDel, del_index


Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
nodes = np.arange(Size)
Seq = get_seq('data/facebook_seq')
Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)

partGraph, luckNodes, swaps = graphPartition(graph=Graph, num=10, k=10, degree=Degree)
# print(type(swaps))                                  # ------------长短不一
# np.savetxt('data/Part_graph', partGraph, delimiter=',')
# np.savetxt('data/Luck_nodes', luckNodes, delimiter=',')
# np.savetxt('data/Swaps', swaps, delimiter=',')

finalGraph, del_index = nodeDelte(graph=partGraph, num=10)
np.savetxt('data/Final_graph', finalGraph, delimiter=',')
np.savetxt('data/Del_index', del_index, delimiter=',')


# partGraph = np.loadtxt('data/Part_graph', delimiter=',')
# luckNodes = np.loadtxt('data/Luck_nodes', delimiter=',')                    # luckNodes 需要增边的结点的索引
# swaps = np.loadtxt('data/swaps', delimiter=',')                             # swaps 若干更新节点的更新边的索引

finalGraph = np.loadtxt('data/Final_graph', delimiter=',')
delIndex = np.loadtxt('data/Del_index', delimiter=',').astype(int)

Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
# nodes = np.arange(Size)
# Seq = get_seq('data/facebook_seq')
# Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)

# Graph = partGraph
nodes = np.arange(Size)
Seq = graphToSeq(Graph)
Degree = np.sum(Graph, axis=1).astype(int)
# ------------ground truth计算-------------------  1612010


ground_resu = louvainClustering(Seq, nodes)
ground_matrix = label_matr_gen(Size, len(ground_resu), ground_resu)
ground_label = label_gen(ground_resu, Size)

G = nxGraphGen(nodes, Seq)
# print(nx.transitivity(G)/3)
# print(nx.average_clustering(G))
# ------------------------计算平均最短路径长度-----------------------------
# shortest = nx.average_shortest_path_length(G)
# print(shortest)


# tri = nx.triangles(G)
# print(np.s um(list(tri.values()))/3)
# print(type(tri.values()))
# print(type(tri))

# cc = CC(G)
# print("clustering coefficient is :", cc)
#
# lis_Q = []
# lis_R = []
# lis_M = []
#

# -------------------------新增节点   delIndex, 直接在原始graph上操作就可以---------------------------------
delta = 1
epsilon = 1

ps = []             # ps代表隐私范围
for i in delIndex:
    print(i)
    se = [j for j in range(Size) if Graph[i, j] == 1]
    ps.append(random_sam(leng=Size, se=se, num=Degree[i]*delta))

p = np.exp(epsilon) / (1 + np.exp(epsilon))
for i in range(len(delIndex)):
    for j in ps[i]:
        if np.random.rand() > p:
            Graph[delIndex[i], j] = 1 - Graph[delIndex[i], j]

print('Node del done!!!!!')


# ------------------------新增边 luckNodes----------------------------------------------------------------
epsilon = 1
Graph = partGraph
k = 10                   # k的值是边更新的阈值
num = len(luckNodes)     # num是更新节点的数量
nums = []                # nums存储luckNodes各自的更新边数量

for i in range(len(swaps)):
    nums.append(len(swaps[i]))
# nums = np.sum(swaps, axis=1)
# for i in range(num):
#     a = np.round(k+np.random.laplace(0, 1))     # a是更新时，新增边的数量。k是需要更新的边的数量
#     if a < 0:
#         a = 0
#     nums.append(a)

    # -----------指定隐私范围------------
privacyScope = []
for i in range(num):
    a = nums[i]
    if a < k:          # 如果真实改变的边数量不够，则需要随机采样一些边填充
        b = np.random.randint(low=0, high=Size, size=k-a)
        c = swaps[i]
        c.extend(b)
        PS = c        # swap[i]存储改节点更新边的索引
    else:
        PS = random.choices(swaps[i], k=a)
    privacyScope.append(PS)

# 对隐私更新边扰动

p = np.exp(epsilon) / (1 + np.exp(epsilon))
for i in range(len(luckNodes)):
    for j in privacyScope[i]:
        if np.random.rand() > p:
            Graph[luckNodes[i], j] = 1-Graph[luckNodes[i], j]

print('Edge del done!!!!!!!')



# for ep in range(1, 10, 1):
#     # -----------------privacy scope 指定----------------------------
#     ps = priSpe(Graph, Degree, delta=1)
#
#     graph1 = Pert(Graph, ps,  epsilon=ep)
#     # nx.triangles(G)
#     Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
#
#     # -------------------graph 恢复+计算序列seq----------------
#     graph1 = graphRecover(graph1)
#     seq1 = graphToSeq(graph1)
#
#     # ----------------计算聚类系数cc和三角数量 ------------------------
#     G = nxGraphGen(nodes, seq1)
#     # G = nxGraphGen(nodes, Seq)
#     tri = nx.triangles(G)
#     print('triangle number is:')
#     print(np.sum(list(tri.values())) / 3)
#     cc = nx.average_clustering(G)
#     print("clustering coefficient is :", cc)
#
#     # -------------计算平均最短路径长度------------------
#     shortest = nx.average_shortest_path_length(G)
#     print('shortest is :')
#     print(shortest)
#
#     # ------------------louvain聚类------------------------
#     resu = louvainClustering(seq1, nodes)
#
#     # ------------------模块度计算--------------------------
#     label_matrix = label_matr_gen(Size, len(resu), resu)
#     Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
#     Q = Modularity(label_matrix, Graph)
#     print('Modularity is:', Q)
#
#     # ----------------------ARI和AMI计算----------------------------
#     label_pre = label_gen(resu, Size)
#     ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
#     ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
#     print('ARI is:', ari_score)
#     print('AMI is:', ami_score)
