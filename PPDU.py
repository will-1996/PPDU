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
import copy

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


def louvainClustering(G):
    # G = nxGraphGen(nodes, seq)
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
    # re = random.choices(re, k=num)
    re = random.sample(re, k=num)
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


def DGG_Gen(Degree):
    """
    利用度序列和一阶零模型生成合成图-------------上限很低  
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


def Degree_pert(deg, epsilon):
    degree = copy.deepcopy(deg)
    for i in range(Size):
        degree[i] = np.round(degree[i]+np.random.laplace(0, 1/epsilon)).astype(int)
        if degree[i] <= 0:
            degree[i] = 1
    return degree


def priSpewithSeq(seq, Size, degree, delta):
    """
    指定所有结点的隐私范围
    :param graph: 原始graph
    :param delta: 隐私倍率
    :return: 隐私范围
    """
    ps = []
    # siz = len(graph)
    degree_p = degree * delta
    for i in range(Size):
        se = seq[i]
        ps.append(random_sam(leng=Size, se=se, num=degree_p[i]))
    return ps


def PPDUwithSeq(seqq, ps, Size, epsilon):
    seq = copy.deepcopy(seqq)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    # p = 1
    for i in range(Size):
        for j in ps[i]:
            if np.random.rand() > p:  # 反转
                if j in seq[i]:
                    seq[i].remove(j)
                else:
                    seq[i].append(j)
                # operate = 0
        # for j in range(Size):
        #     if j in ps:  # e_ij属于隐私范围
        #         if j in seq:  # e_ij=1
        #             if np.random.rand() > p:  # 反转
        #                 seq[i].remove(j)
        #             # else:
        #             #     pass
        #         else:  # e_ij属于隐私范围且 e_ij=0
        #             if np.random.rand() < p:
        #                 seq[i].append(j)  # ----------------- 这里将seq 的顺序打乱了
            #         else:
            #             pass
            # else:  # e_ij不属于隐私范围，不做处理
            #     pass
    return seq

Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
nodes = np.arange(Size)
Seq = get_seq('data/facebook_seq')

Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)


# ------------ground truth计算-------------------  1612010

# Seq = graphToSeq(Graph)
G = nxGraphGen(nodes, Seq)
ground_resu = louvainClustering(G)
ground_matrix = label_matr_gen(Size, len(ground_resu), ground_resu)
ground_label = label_gen(ground_resu, Size)


ground_glo_cc = nx.transitivity(G)/3
ground_cc = nx.average_clustering(G)
# ground_shor_path = nx.average_shortest_path_length(G)
# print(nx.transitivity(G)/3)
# print(nx.average_clustering(G))
# resu = louvainClustering(Seq, nodes)

# ------------------模块度计算--------------------------
# Q = nx.algorithms.community.modularity(G, resu)

# a= nx.modularity_spectrum(G)

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

def full_privacy(size):
    ps = []
    for i in range(size):
        ps.append(list(range(size)))
        ps[i].remove(i)
    return ps

# a = full_privacy(5)
# print(a)


# for ep in range(1, 9, 1):
#     print('----------------------epsilon is :', ep)
#     # -----------------privacy scope 指定----------------------------
#     # ps = priSpe(Graph, Degree, delta=4000)
#     ps = full_privacy(4039)
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
#     print('Global clustering coefficient error is:')
#     print(1-(nx.transitivity(G)/3)/ground_glo_cc)
#     print('CC error is:')
#     cc = nx.average_clustering(G)
#     print(1-cc/ground_cc)
#     # print('triangle number is:')
#     # print(np.sum(list(tri.values())) / 3)
#     # print("clustering coefficient is :", cc)
#
#     # -------------计算平均最短路径长度------------------
#     # shortest = nx.average_shortest_path_length(G)
#     # print('shortest path error is :')
#     # print(1-shortest/ground_shor_path)
#
#     # ------------------louvain聚类------------------------
#     resu = louvainClustering(G)
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


for ep in range(8, 9, 1):
    print('----------------------epsilon is :', ep)

    # -----------------privacy scope 指定----------------------------
    ps = priSpewithSeq(seq=Seq, Size=Size, degree=Degree, delta=1)

    # ps = full_privacy(4039)
    # -------------------graph 恢复+计算序列seq----------------
    seq1 = PPDUwithSeq(seqq=Seq, ps=ps, Size=Size, epsilon=ep)
    # for j in range(Size):
    #     print(len(seq1[j])-Degree[j])


    # print('sequence generated!')
    # ----------------计算聚类系数cc和三角数量 ------------------------
    G = nxGraphGen(nodes, seq1)
    # tri = nx.triangles(G)
    print('Global clustering coefficient error is:')
    print(1-(nx.transitivity(G)/3)/ground_glo_cc)
    print('CC error is:')
    cc = nx.average_clustering(G)
    print(1-cc/ground_cc)

    # -------------计算平均最短路径长度------------------
    # shortest = nx.average_shortest_path_length(G)
    # print('shortest path error is :')
    # print(1-shortest/ground_shor_path)

    # ------------------louvain聚类------------------------
    resu = louvainClustering(G)

    # ------------------模块度计算--------------------------
    Q = nx.algorithms.community.modularity(G, resu)
    print('Modularity is:', Q)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu, Size)
    ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)



label = np.zeros(4039)

for i in range(len(resu)):
    for j in range(len(resu[i])):
        label[resu[i][j]] = i
resu_name = 'ep_8_result.txt'

np.savetxt(resu_name, label, fmt='%d', delimiter=',')

# ------------------------RABV------------------------------------
# Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
# for ep in range(1, 9, 1):
#     print('--------------New round start, epsilon=', ep)
#     graph2 = Random_R(graph=Graph, epsilon=ep)
#             # ------ 这里采用两倍隐私预算来模拟rabv的效果，另外最终graph根据邻接矩阵右上角的取值确定
#     for i in range(Size):
#         graph2[i, i] = 0      # 对角线置为0，消除self-loop
#         for j in range(i+1, Size):
#             graph2[j, i] = graph2[i, j]
#     seq2 = graphToSeq(graph2)
#
#     # ----------------计算聚类系数cc和三角数量 ------------------------
#     G = nxGraphGen(nodes, seq2)
#     # G = nxGraphGen(nodes, Seq)
#     tri = nx.triangles(G)
#     print('Global clustering coefficient error is:')
#     print(1 - (nx.transitivity(G) / 3) / ground_glo_cc)
#     print('CC error is:')
#     cc = nx.average_clustering(G)
#     print(1 - cc / ground_cc)
#     # print('triangle number is:')
#     # print(np.sum(list(tri.values())) / 3)
#     # print("clustering coefficient is :", cc)
#
#     # -------------计算平均最短路径长度------------------
#     # shortest = nx.average_shortest_path_length(G)
#     # print('shortest path error is :')
#     # print(1 - shortest / ground_shor_path)
#
#     # ------------------louvain聚类------------------------
#     resu = louvainClustering(G)
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



# ----------------计算聚类系数cc和三角数量 ------------------------
# for ep in range(3, 9):
#     print('/n')
#     print('----------------------epsilon is :', ep)
#     degree = Degree_pert(Degree, ep)
#     graph3 = DGG_Gen(Degree=degree)
#     seq3 = graphToSeq(graph3)
#     G = nxGraphGen(nodes, seq3)
#     # G = nxGraphGen(nodes, Seq)
#     tri = nx.triangles(G)
#     print('Global clustering coefficient error is:')
#     print(1 - (nx.transitivity(G) / 3) / ground_glo_cc)
#     print('CC error is:')
#     cc = nx.average_clustering(G)
#     print(1 - cc / ground_cc)
#     # print('triangle number is:')
#     # print(np.sum(list(tri.values())) / 3)
#     # print("clustering coefficient is :", cc)
#
#     # -------------计算平均最短路径长度------------------
#     shortest = nx.average_shortest_path_length(G)
#     print('shortest path error is :')
#     print(1 - shortest / ground_shor_path)
#
#     # ------------------louvain聚类------------------------
#     resu = louvainClustering(seq3, nodes)
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

# seq1 = graphToSeq(Graph)
# print(seq1[0])

# for Ep in range(1, 9):
#     # Ep = Ep/10
#     # ---------------graph扰动及Sequence计算-------------------
#     # graph1 = rabv(graph=Graph, epsilon=Ep)                 # RABV-only, baseline方法
#     graph1 = graphPert(graph=Graph, epsilon=Ep)         # 不考虑不存在的边
#     graph1 = graphPertPro(graph=Graph, epsilon=Ep, fake=4, degree=Degree)
#     graph1 = graphRecover(graph1)
#     seq1 = graphToSeq(graph1)


#     # ---------------------louvain聚类----------------------------
#     resu = louvainClustering(seq1, nodes)
#
#     # -----------------------模块度计算----------------------------
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
# 
#     lis_Q.append(Q)
#     lis_R.append(ari_score)
#     lis_M.append(ami_score)
# 
# print(lis_Q)
# print(lis_R)
# print(lis_M)







