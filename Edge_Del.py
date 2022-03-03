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
    Nd.sort(reverse=True)
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
    re = list(set(seq) - set(se))
    # 从non-existing edges中采样privacy edges
    re = random.sample(re, k=num)
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


def Node_update(sseq, nd, eps, nd_seq, num):
    """
    生成节点添加的 新图
    :param seq:
    :param nd:
    :param eps:
    :param nd_seq:  节点的ground_truth 边情况
    :param num:
    :return:
    """
    seq = copy.deepcopy(sseq)
    size = len(seq)
    p = np.exp(eps) / (1 + np.exp(eps))
    for i in range(len(nd)):
        # 为每个节点指定隐私范围
        private_scope = random_sam(leng=size, se=nd_seq[i], num=num*len(nd_seq))
        for j in private_scope:
            # 为隐私范围的每个节点的连接信息进行扰动
            if np.random.rand() > p:  # 反转
                if j in nd_seq[i]:
                    nd_seq[i].remove(j)
                else:
                    nd_seq[i].append(j)
        # 将扰动过的边信息 并入 残缺图中
        seq[nd[i]] = nd_seq[i]

    return seq


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


def node_sample(degrees, num_edge, num_node):       # 随机选择符合条件的删边节点
    # degrees = 0
    # num_edge = 1000
    # num_node = 100
    avg = num_edge//num_node
    large_nodes = []
    for i in range(len(degrees)):
        if degrees[i] > avg:
            large_nodes.append(i)

    Nd = random.sample(large_nodes, num_node)
    Nd.sort()           # Nd 是删边节点的索引 组成的数组，  已排序
    return Nd


def graph_edge_del(sseq, nd, num_edge, num_node):
    # -----------------传入的seq未被改变--------------------------
    seq = copy.deepcopy(sseq)
    avg = num_edge // num_node
    dels = []           # 保存删除的边的集合
    for i in range(len(nd)):
        del_ind = random.sample(sseq[nd[i]], avg)
        dels.append(del_ind)
        # 从该节点的边序列中 删除所有 应删除的边
        seq[nd[i]] = list(set(seq[nd[i]])-set(del_ind))
        for j in del_ind:       # 另一个端点的边信息也应该删除
            # del_ind 节点i删除的边序列     nd 进行操作的节点序列
            if nd[i] in seq[j]:
                seq[j].remove(nd[i])
    return seq, dels


def k_rr(seq, epsilon, dele):
    """
    单个节点的krr过程， 返回扰动之后 用户提交的 边
    :param seq:   节点 i 的 连接序列
    :param epsilon:
    :param dele:  被删除的边的集合
    :return:
    """
    avg = len(dele)
    k = len(seq)
    p = np.exp(epsilon)/(k-1+np.exp(epsilon))
    fake = 0
    for i in range(avg):
        if np.random.rand() > p:        # 隐藏真实值
            fake += 1
    # count 个虚拟值， avg-count个真实值, avg: 每个节点更新边的数量
    real = avg -fake
    reals= random.sample(dele, real)
    left = list(set(seq)-set(reals))
    fakes = random.sample(left, fake)
    fakes.extend(reals)
    fakes.sort()
    return fakes


def edge_recover(nd, deleted_seq, dels, seq, epsilon, node_num):
    # deleted_seq 是t0时刻graph的真实状态， dels是该时刻边的真实更新状态， fake_seq 是该时刻节点们提交的信息， 节点的序号存储于nd中

    # # 计算删除节点的索引
    # nd = node_sample(degrees=degrees, num_edge=edge_num, num_node=node_num)
    # # 先构建 删除边之后的残缺图， 并获取具体删除边的信息
    # deleted_seq, dels = graph_edge_del(sseq=seq, nd=nd, num_edge=edge_num, num_node=node_num)

    fake_seq = []
    for i in range(node_num):
        # ---------- 计算各个用户提交的 删边信息 --------------------
        fakes = k_rr(seq=seq[nd[i]], epsilon=epsilon, dele=dels[i])
        fake_seq.append(fakes)

    for i in range(node_num):       # 利用用户提交的加噪 边更新信息 加入到 残缺图中
        deleted_seq[nd[i]].extend(fake_seq[i])
        deleted_seq[nd[i]].sort()

    return deleted_seq


def graph_del(nd, dels, seq, epsilon, node_num):
    # deleted_seq 是t1时刻graph的真实状态， dels是该时刻边的真实更新状态， fake_seq 是该时刻节点们提交的信息， 节点的序号存储于nd中
    # # 计算删除节点的索引
    # nd = node_sample(degrees=degrees, num_edge=edge_num, num_node=node_num)
    # # 先构建 删除边之后的残缺图， 并获取具体删除边的信息
    # deleted_seq, dels = graph_edge_del(sseq=seq, nd=nd, num_edge=edge_num, num_node=node_num)
    noisy_seq = copy.deepcopy(seq)
    fake_seq = []
    for i in range(node_num):
        # ---------- 计算各个用户提交的 删边信息 --------------------
        fakes = k_rr(seq=noisy_seq[nd[i]], epsilon=epsilon, dele=dels[i])
        fake_seq.append(fakes)

    for i in range(node_num):       # 利用用户提交的加噪 边更新信息 加入到 残缺图中
        noisy_seq[nd[i]] = list(set(noisy_seq[nd[i]])-set(fake_seq[i]))
        noisy_seq[nd[i]].sort()

    return noisy_seq


def RNL_disturb(seq, seq_del, epsilon):
    # 仅处理了 当前节点的边索引， 未对 另一端的好友节点的边索引进行处理。
    # num为真实删除边的数量, seq_del 真实删除边的索引
    p = np.exp(epsilon) / (1 + np.exp(epsilon))

    sseq = copy.deepcopy(seq)
    # 反转但 真的被删除的  会保持存在， 即 既属于 s， 又属于 seq_del，
    # 其余边， 反转则删除
    for i in range(len(seq)):
        if np.random.rand() > p:
            if seq[i] not in seq_del:
                sseq.remove(seq[i])     # 反转且不在删除边中， 删除该边
        else:
            if seq[i] in seq_del:
                sseq.remove(seq[i])     # 不反转，但属于删除边，删除该边
    return sseq



# Graph = np.loadtxt('data/Facebook_graph', delimiter=',')
nodes = np.arange(Size)
Seq = get_seq('data/facebook_seq')

Degree = np.loadtxt('data/facebook_degree', delimiter=',').astype(int)

# ------------ground truth计算-------------------  1612010

# Seq = graphToSeq(Graph)
G = nxGraphGen(nodes, Seq)
ground_resu = louvainClustering(G)
# ground_matrix = label_matr_gen(Size, len(ground_resu), ground_resu)
ground_label = label_gen(ground_resu, Size)


ground_glo_cc = nx.transitivity(G)/3
# print('global cc is:', ground_glo_cc)
ground_cc = nx.average_clustering(G)
# print('cc is', ground_cc)

edge_num = 10000
node_num = 100

# total = 0
# for i in range(Size):
#     total += len(Seq[i])
#
# print(total)

# 被删除边的节点的索引
nd = node_sample(degrees=Degree, num_edge=edge_num, num_node=node_num)
# 残缺图 和 真正删除的边信息
deleted_seq, dels = graph_edge_del(sseq=Seq, nd=nd, num_edge=edge_num, num_node=node_num)

# count = 0
# for i in range(Size):
#     count += len(deleted_seq[i])
#
# print(count)

# ----------------   计算 残缺图的各种指标 -------------
G_deleted = nxGraphGen(nodes, deleted_seq)
resu_deleted = louvainClustering(G_deleted)

# glo_cc_noisy = nx.transitivity(G_noisy)/3
print('t1 Global clustering coefficient is:')
glo_cc_deleted = nx.transitivity(G_deleted)/3
print(glo_cc_deleted)

print('t1 CC is:')
cc_deleted = nx.average_clustering(G_deleted)
print(cc_deleted)

Q_deleted = nx.algorithms.community.modularity(G_deleted, resu_deleted)
print('t1 Modularity is:', Q_deleted)

# ----------------------ARI和AMI计算----------------------------
label_pre_deleted = label_gen(resu_deleted, Size)
# ari_score = metrics.adjusted_rand_score(ground_label, label_pre_deleted)
# ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre_deleted)
# print('ARI is:', ari_score)
# print('AMI is:', ami_score)


# -----------------  计算  合成图的各种指标 -------------------
for eps in range(1, 9, 1):
    print('privacy budget is:', eps)
    # new_seq = edge_recover(nd=nd, deleted_seq=deleted_seq, dels=dels, seq=Seq, epsilon=0.01, node_num=node_num)
    new_seq = graph_del(nd=nd, dels=dels, seq=Seq, epsilon=eps, node_num=node_num)

    G_noisy = nxGraphGen(nodes, new_seq)
    resu_noisy = louvainClustering(G_noisy)

    glo_cc_noisy = nx.transitivity(G_noisy)/3
    print('Global clustering coefficient error is:')
    print(np.abs(glo_cc_noisy-glo_cc_deleted)/glo_cc_deleted)

    print('CC error is:')
    cc_noisy = nx.average_clustering(G_noisy)
    print(np.abs(cc_noisy-cc_deleted)/cc_deleted)


    Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
    print('Modularity is:', Q_noisy)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu_noisy, Size)
    ari_score = metrics.adjusted_rand_score(label_pre_deleted, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(label_pre_deleted, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)


#  ----------------------------DGG --------------------------
# avg = edge_num//node_num
# for eps in range(1, 9, 1):
#     print('epsilon is : ----------------------------------', eps)
#     sseq = copy.deepcopy(Seq)
#     noisy_degree_del = []  # 存储用户提交的加噪度值
#     for j in range(node_num):
#         noisy_degree_del.append(np.abs(np.round(avg+np.random.laplace(loc=0, scale=1/eps))))
#
#     noisy_degree_del = np.array(noisy_degree_del).astype(int)
#     # 从已经存在的seq中随机删除 noisy del个边
#     for j in range(node_num):
#
#         if noisy_degree_del[j] > len(sseq[nd[j]]):           # 提交度值大于本身度值的情况
#             noisy_degree_del[j] = len(sseq[nd[j]])
#
#         del_ind_j = random.sample(sseq[nd[j]], noisy_degree_del[j])
#         sseq[nd[j]] = list(set(sseq[nd[j]])-set(del_ind_j))
#         for k in del_ind_j:             # 删除另一端的边信息
#             if nd[j] in sseq[nd[j]]:
#                 sseq[nd[j]].remove(nd[j])
#
#     new_seq = sseq
#
#     G_noisy = nxGraphGen(nodes, new_seq)
#     resu_noisy = louvainClustering(G_noisy)
#
#     glo_cc_noisy = nx.transitivity(G_noisy) / 3
#     print('Global clustering coefficient error is:')
#     print(np.abs(glo_cc_noisy - glo_cc_deleted) / glo_cc_deleted)
#
#     print('CC error is:')
#     cc_noisy = nx.average_clustering(G_noisy)
#     print(np.abs(cc_noisy - cc_deleted) / cc_deleted)
#
#     Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
#     print('Modularity is:', Q_noisy)
#
#     # ----------------------ARI和AMI计算----------------------------
#     label_pre = label_gen(resu_noisy, Size)
#     ari_score = metrics.adjusted_rand_score(label_pre_deleted, label_pre)
#     ami_score = metrics.adjusted_mutual_info_score(label_pre_deleted, label_pre)
#     print('ARI is:', ari_score)
#     print('AMI is:', ami_score)



# -----------------------RNL ----------------------------------
avg = edge_num//node_num
for eps in range(1, 9, 1):
    print('epsilon is : ----------------------------------', eps)
    sseq = copy.deepcopy(Seq)
    for j in range(node_num):
        noisy_seq_j = RNL_disturb(seq=sseq[nd[j]], seq_del=dels[j], epsilon=eps)
        sseq[nd[j]] = noisy_seq_j

    new_seq = sseq
    G_noisy = nxGraphGen(nodes, new_seq)
    resu_noisy = louvainClustering(G_noisy)

    glo_cc_noisy = nx.transitivity(G_noisy) / 3
    print('Global clustering coefficient error is:')
    print(np.abs(glo_cc_noisy - glo_cc_deleted) / glo_cc_deleted)

    print('CC error is:')
    cc_noisy = nx.average_clustering(G_noisy)
    print(np.abs(cc_noisy - cc_deleted) / cc_deleted)

    Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
    print('Modularity is:', Q_noisy)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu_noisy, Size)
    ari_score = metrics.adjusted_rand_score(label_pre_deleted, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(label_pre_deleted, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)
