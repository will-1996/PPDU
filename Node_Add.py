import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import metrics
import random
import copy

# facebook=4039     enron=36692        astro=18772          gplus=107614
Size = 4039
# Ep = 1

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

def Node_delete(num):
    """
    随机选取删除的节点序号， 以逆序返回，方便下一步的删除操作
    :param num:
    :return: Nd 删除结点的索引号， 逆序
    """
    inde = range(Size)
    Nd = random.sample(inde, num)
    Nd.sort(reverse=1)
    # Nd = np.random.randint(low=0, high=100, size=10)
    return Nd


def NewSeq(Seq, Nd):
    # 生成 删除节点信息后的残缺图
    Nd_seq = []         # 将删除结点的 边信息备份， 逆序
    # ---------------------删除节点=删除该结点的所有边信息-------------------------
    # 图的矩阵形状维持不变，只是将删除结点的边信息置空
    for i in Nd:
        Nd_seq.append(Seq[i])
        Seq[i] = []
    # Size = len(Seq)
    for i in range(Size):
        Seq[i] = list(set(Seq[i])-set(Nd))
    return Seq, Nd_seq



def random_sam(leng, se, num):
    """
    为单个用户指定隐私范围
    :param leng: graph size n
    :param se: IDs of existing edges
    :param num: delta--parameter of privacy scope
    :return:  privacy scope
    """
    seq = list(range(leng))
    if num + len(se) >= leng:
        return seq
    re = list(set(seq) - set(se))       # non-existing 边的索引集合
    # 从non-existing edges中采样privacy edges
    re = random.choices(re, k=num)
    # 将existing edges加入privacy edges
    re.extend(se)
    # 对privacy scope排序
    re.sort()
    return re


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


def Node_update(sseq, nd, eps, nnd_seq, num):
    """
    生成节点添加的 新图
    :param sseq: 残缺图 ground truth的边序列
    :param nd: 新增节点的索引
    :param eps: 隐私预算
    :param nd_seq:  新增节点的ground_truth 边情况
    :param num: 隐私范围参数
    :return:
    """
    seq = copy.deepcopy(sseq)
    nd_seq = copy.deepcopy(nnd_seq)
    size = 4039
    p = np.exp(eps) / (1 + np.exp(eps))
    # p=0.99
    for i in range(len(nd)):
        # 为每个节点指定隐私范围
        private_scope = random_sam(leng=size, se=nd_seq[i], num=num*len(nd_seq))
        for j in private_scope:
            # 为隐私范围的每个节点的连接信息进行扰动
            if np.random.rand() > p:  # 反转
                if j in nd_seq[i]:          # 原本为1， 再反转， 则删除
                    nd_seq[i].remove(j)
                else:
                    nd_seq[i].append(j)
        # 将扰动过的边信息 并入 残缺图中
        seq[nd[i]] = nd_seq[i]

    return seq



def Node_mul_update(times, sseq, nd, eps, nnd_seq, num):
    sq = copy.deepcopy(sseq)
    eps = eps/times     # 平分隐私预算
    total_nodes = len(nd)       #
    each_nodes = total_nodes/times      # 计算每次更新的节点数量
    for s in range(1, times+1, 1):              # 进行给定次数的节点添加
        # print(int(each_nodes*s))
        sq = Node_update(sseq=sq, nd=nd[int(each_nodes*(s-1)):int(each_nodes*s)], eps=eps, nnd_seq=nnd_seq[int(each_nodes*(s-1)):int(each_nodes*s)], num=num)
    return sq

def label_gen(re, n):
    label = np.zeros(n)
    for c in range(len(re)):
        for i in re[c]:
            label[i] = c
    return label


def label_matr_gen(node_num, clu_num, res):
    label_matrix = np.zeros([node_num, clu_num])
    for i in range(len(res)):
        for j in res[i]:
            label_matrix[int(j)][i] = 1
    return label_matrix


def get_seq(read_path):
    f = open(read_path, 'r')
    seq = []
    for line in f:
        a = line.strip().split(',')
        a = [int(i) for i in a]
        # map(int, a)
        seq.append(a)
    return seq


def RNL_pert(seq, size, epsilon):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    setq = set(seq)
    s = set()  # 需要反转的边的索引， 1->0 & 0->1
    rands = np.random.rand(size)
    rands = np.ceil(rands - p)  # 若rand 大于p则rans=1，反转
    for j in range(len(rands)):
        if rands[j]:
            s.add(j)
    inte = setq.intersection(s)  # 计算反转和 存在边的交集，即 应该 1->0的边
    setq.update(s)  # 计算 s和 setq的并集
    # 并集-交集  留下的部分为 不同时出现在两个集合中的元素，即现存但不反转，or 反转但现在没有
    x = list(setq - inte)  # 反转边的集合减存在边的集合，若反转且存在，则不存在。若反转但不存在，则存在。
    return x





# 根据 度值信息预测边的连接


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

# ----------------------计算 残缺图的各种指标 --------------------------
node_del_num = 500                              # 删除节点的数量

nd = Node_delete(node_del_num)
seq_del, nd_seq = NewSeq(Seq=Seq, Nd=nd)        # seq_Del, 残缺图 nd_seq, 删点的边
# G_del = nxGraphGen(nodes, seq_del)
# resu_del = louvainClustering(G_del)
# # del_matrix = label_matr_gen(Size, len(resu_del), resu_del)
# # del_label = label_gen(resu_del, Size)
# glo_cc_del = nx.transitivity(G_del)/3
# cc_del = nx.average_clustering(G_del)
# Q_del = nx.algorithms.community.modularity(G_del, resu_del)


# ---------------------PPDU  ----计算 扰动图的各种指标 --------------------------------
for eps in range(1, 9, 1):
    print('privacy budget is:', eps)
    seq_noisy = Node_mul_update(times=4, sseq=seq_del, nd=nd, eps=eps, nnd_seq=nd_seq, num=1)
    # seq_noisy = Node_update(sseq=seq_del, nd=nd, eps=eps, nnd_seq=nd_seq, num=1)       # num,隐私范围参数
    G_noisy = nxGraphGen(nodes, seq_noisy)
    resu_noisy = louvainClustering(G_noisy)
    # noisy_matrix = label_matr_gen(Size, len(resu_noisy), resu_noisy)
    # noisy_label = label_gen(resu_noisy, Size)
    glo_cc_noisy = nx.transitivity(G_noisy)/3
    print('global cc error is:', np.abs(glo_cc_noisy-ground_glo_cc)/ground_glo_cc)
    cc_noisy = nx.average_clustering(G_noisy)
    print('average cc error is:', np.abs(cc_noisy-ground_cc)/ground_cc)

    Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
    print('Modularity is:', Q_noisy)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu_noisy, Size)
    ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)


#    ---------------------------DGG -------------------
#  提交 新增节点的度值，根据其余节点的度值作为权重来随机选取 连接边
# for eps in range(1, 9, 1):
#     print('epsilon is : ----------------------------------', eps)
#     seq_noisy = copy.deepcopy(seq_del)
#     # eps = 1
#     noisy_degree_add = []           # 存储用户提交的加噪度值
#     for j in range(node_del_num):
#         noisy_degree_add.append(np.abs(np.round(Degree[nd[j]]+np.random.laplace(loc=0, scale=1/eps))))
#
#     noisy_degree_add = np.array(noisy_degree_add).astype(int)
#
#     #  --------------------- 根据加噪度向量， 对graph进行更新--------------------
#     p = Degree/np.sum(Degree)
#     inds = np.array(list(range(Size))).astype(int)
#     for j in range(node_del_num):
#         seq_j = list(np.random.choice(a=inds, size=noisy_degree_add[j], replace=False, p=p))
#         if nd[j] in seq_j:
#             seq_j.remove(nd[j])                 # 删除可能的self-loop
#         seq_noisy[nd[j]] = seq_j          # 这里改变了残缺图的信息， 在迭代时需要注意
#
#     G_noisy = nxGraphGen(nodes, seq_noisy)
#     resu_noisy = louvainClustering(G_noisy)
#     # noisy_matrix = label_matr_gen(Size, len(resu_noisy), resu_noisy)
#     # noisy_label = label_gen(resu_noisy, Size)
#     glo_cc_noisy = nx.transitivity(G_noisy) / 3
#     print('global cc error is:', np.abs(glo_cc_noisy - ground_glo_cc) / ground_glo_cc)
#     cc_noisy = nx.average_clustering(G_noisy)
#     print('average cc error is:', np.abs(cc_noisy - ground_cc) / ground_cc)
#
#     Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
#     print('Modularity is:', Q_noisy)
#
#     # ----------------------ARI和AMI计算----------------------------
#     label_pre = label_gen(resu_noisy, Size)
#     ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
#     ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
#     print('ARI is:', ari_score)
#     print('AMI is:', ami_score)


#    ---------------------------RNL -------------------
#     节点直接对边信息进行RR扰动，然后并入graph中

# for eps in range(1, 9, 1):
#     print('epsilon is : ----------------------------------', eps)
#     seq_noisy = copy.deepcopy(seq_del)
#
#     seq_del, nd_seq = NewSeq(Seq=Seq, Nd=nd)  # seq_Del, 残缺图 nd_seq, 删点的边
#
#     times = 8
#     eps = eps/times
#     for j in range(node_del_num):
#
#         noisy_seq_j = RNL_pert(seq=nd_seq[j], size=Size, epsilon=eps)
#         seq_noisy[nd[j]] = noisy_seq_j  # 这里改变了残缺图的信息， 在迭代时需要注意
#
#     G_noisy = nxGraphGen(nodes, seq_noisy)
#     resu_noisy = louvainClustering(G_noisy)
#     # noisy_matrix = label_matr_gen(Size, len(resu_noisy), resu_noisy)
#     # noisy_label = label_gen(resu_noisy, Size)
#     glo_cc_noisy = nx.transitivity(G_noisy) / 3
#     print('global cc error is:', np.abs(glo_cc_noisy - ground_glo_cc) / ground_glo_cc)
#     cc_noisy = nx.average_clustering(G_noisy)
#     print('average cc error is:', np.abs(cc_noisy - ground_cc) / ground_cc)
#
#     Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
#     print('Modularity is:', Q_noisy)
#
#     # ----------------------ARI和AMI计算----------------------------
#     label_pre = label_gen(resu_noisy, Size)
#     ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
#     ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
#     print('ARI is:', ari_score)
#     print('AMI is:', ami_score)



 


























