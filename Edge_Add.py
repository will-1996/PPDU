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
    p = 0
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



def RNL_rr(left_seq, seq_add, epsilon, size):
    nonExi_seq = list(range(size))              # 所有目前不存在的边的集合
    nonExi_seq = list(set(nonExi_seq)-set(left_seq))
    p = np.exp(epsilon) / (1 + np.exp(epsilon))

    noi_seq = copy.deepcopy(left_seq)       # left_seq代表删除之后，还幸存的边的集合
    for i in range(len(nonExi_seq)):            # noi_seq
        if np.random.rand() < p:            # 不需要反转
            if nonExi_seq[i] in seq_add:        # seq_add 代表真实新增边的集合
                noi_seq.append(nonExi_seq[i])       # 保持 且属于新增的边， 则新增该边
            # else:
            #     noi_seq.remove(i)
        else:                           # 需要反转
            if nonExi_seq[i] not in seq_add:
                noi_seq.append(nonExi_seq[i])       # 反转 且不属于新增边， 新增该边

    noi_seq.sort()
    return noi_seq


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
print('global cc is:', ground_glo_cc)
ground_cc = nx.average_clustering(G)
print('cc is', ground_cc)

edge_num = 10000
node_num = 100
# epsilon = 1
nd = node_sample(degrees=Degree, num_edge=edge_num, num_node=node_num)
# 残缺图 以及 新增边的集合
ddeleted_seq, dels = graph_edge_del(sseq=Seq, nd=nd, num_edge=edge_num, num_node=node_num)

# ----------------   计算 残缺图的各种指标 -------------
# GG_deleted = nxGraphGen(nodes, ddeleted_seq)

# nd_1_bunch = []
# for i in range(4039):
#     nd_1_bunch.append((nd[0], i))
# nd_1_bunch.remove((nd[0], nd[0]))
#
# prefer_attach = nx.preferential_attachment(G=G_deleted, ebunch=nd_1_bunch)
#
# pa = []
# for i in range(4039):
#     pa.append(0)
#
# for u, v, p in prefer_attach:
#     print(f"({u}, {v}) -> {p}")
#     pa[v] = p
#
# pa = np.array(pa)
# a = pa.argsort()
avg = edge_num//node_num

# ----------------------------PPDU  提交扰动增边数+链接预测---------------------------
# for eps in range(1, 9, 1):
#
#
#     # G_deleted = copy.deepcopy(GG_deleted)
#     deleted_seq = copy.deepcopy(ddeleted_seq)
#     G_deleted = nxGraphGen(nodes, deleted_seq)
#     print('privacy budget is:', eps)
#     noisy_num = []
#
#     times = 50
#     eps = eps / times
#
#     for i in range(node_num):
#         noisy_num.append(np.round(avg+np.random.laplace(0, 1/eps)).astype(int))
#
#     # count = 0
#     for i in range(node_num):
#
#         nd_1_bunch = []
#         for j in range(4039):
#             nd_1_bunch.append((nd[i], j))
#         nd_1_bunch.remove((nd[i], nd[i]))
#         # prefer_attach = nx.preferential_attachment(G=G_deleted, ebunch=nd_1_bunch)
#         # jaccard = nx.preferential_attachment(G=G_deleted, ebunch=nd_1_bunch)
#
#         adar = nx.adamic_adar_index(G=G_deleted, ebunch=nd_1_bunch)
#
#         pa = []         # pa: 存储节点i 相似度的列表
#         for j in range(4039):
#             pa.append(0)
#
#         # for u, v, p in prefer_attach:
#         #     # print(f"({u}, {v}) -> {p}")
#         #     pa[v] = p
#         for u, v, p in adar:
#             # print(f"({u}, {v}) -> {p}")
#             pa[v] = p
#
#
#         pa = np.array(pa)
#         # 将 已经残缺图中 相连的边 的相似度置为 0
#         for j in deleted_seq[nd[i]]:
#             pa[j] = 0
#         a = pa.argsort()
#         # 想残缺图中 加入noisy数量的 预测边
#         deleted_seq[nd[i]] = list(set(deleted_seq[nd[i]])|set(a[-noisy_num[i]:]))
#         deleted_seq[nd[i]].sort()
#     #     b = list(set(a[-avg:])&set(dels[i]))
#     #     count += len(b)
#     #
#     # print(count)
#
#     G_deleted = nxGraphGen(nodes, deleted_seq)
#     resu_deleted = louvainClustering(G_deleted)
#
#     # glo_cc_noisy = nx.transitivity(G_noisy)/3
#     print('Global clustering coefficient error is:')
#     print(1-(nx.transitivity(G_deleted)/3)/ground_glo_cc)
#
#     print('CC error is:')
#     cc_deleted = nx.average_clustering(G_deleted)
#     print(1-cc_deleted/ground_cc)
#
#     Q_deleted = nx.algorithms.community.modularity(G_deleted, resu_deleted)
#     print('Modularity is:', Q_deleted)
#
#     # ----------------------ARI和AMI计算----------------------------
#     label_pre = label_gen(resu_deleted, Size)
#     ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
#     ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
#     print('ARI is:', ari_score)
#     print('AMI is:', ami_score)
#     deleted_seq = []



# ------------------DGG  提交扰动增边数，根据度值确定连接----------------------------
# for eps in range(1, 9, 1):
#     # G_deleted = copy.deepcopy(GG_deleted)
#     deleted_seq = copy.deepcopy(ddeleted_seq)
#
#     print('privacy budget is:', eps)
#     noisy_num = []              # 扰动后的增边数量
#     for i in range(node_num):
#         noisy_num.append(np.round(avg+np.random.laplace(0, 1/eps)).astype(int))
#
#     deg = copy.deepcopy(Degree)
#     for j in nd:
#         deg[j] = 0              # 将更新节点的连边概率置为0， 合理？？？
#     p = deg/np.sum(deg)
#     inds = np.array(list(range(Size))).astype(int)
#
#     for j in range(node_num):
#         seq_j = list(np.random.choice(a=inds, size=noisy_num[j], replace=False, p=p))
#         # if nd[j] in seq_j:
#         #     seq_j.remove(nd[j])                 # 删除可能的self-loop
#         deleted_seq[nd[j]] = seq_j          # 这里改变了残缺图的信息， 在迭代时需要注意
#
#     G_noisy = nxGraphGen(nodes, deleted_seq)
#     resu_noisy = louvainClustering(G_noisy)
#     # noisy_matrix = label_matr_gen(Size, len(resu_noisy), resu_noisy)
#     # noisy_label = label_gen(resu_noisy, Size)
#     glo_cc_noisy = nx.transitivity(G_noisy) / 3
#     print('global cc error is:', np.abs(glo_cc_noisy - ground_glo_cc) / ground_cc)
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






# -----------------------RNL ---------------------------------- ddeleted_seq, dels

for eps in range(1, 9, 1):
    print('epsilon is : ----------------------------------', eps)
    sseq = copy.deepcopy(ddeleted_seq)
    times = 8
    eps = eps/times
    for j in range(node_num):
        noisy_seq_j = RNL_rr(left_seq=sseq[nd[j]], seq_add=dels[j], epsilon=eps, size=Size)
        sseq[nd[j]] = noisy_seq_j

    new_seq = sseq
    G_noisy = nxGraphGen(nodes, new_seq)
    resu_noisy = louvainClustering(G_noisy)

    glo_cc_noisy = nx.transitivity(G_noisy) / 3
    print('Global clustering coefficient error is:')
    print(np.abs(glo_cc_noisy - ground_glo_cc) / ground_glo_cc)

    print('CC error is:')
    cc_noisy = nx.average_clustering(G_noisy)
    print(np.abs(cc_noisy - ground_cc) / ground_cc)

    Q_noisy = nx.algorithms.community.modularity(G_noisy, resu_noisy)
    print('Modularity is:', Q_noisy)

    # ----------------------ARI和AMI计算----------------------------
    label_pre = label_gen(resu_noisy, Size)
    ari_score = metrics.adjusted_rand_score(ground_label, label_pre)
    ami_score = metrics.adjusted_mutual_info_score(ground_label, label_pre)
    print('ARI is:', ari_score)
    print('AMI is:', ami_score)







# -----------------  计算  合成图的各种指标 -------------------
# for eps in range(1, 9, 1):
#     print('privacy budget is:', eps)
#
#     new_seq = edge_recover(nd=nd, deleted_seq=ddeleted_seq, dels=dels, seq=Seq, epsilon=eps, node_num=node_num)
#
#     G_noisy = nxGraphGen(nodes, new_seq)
#     resu_noisy = louvainClustering(G_noisy)
#
#     glo_cc_noisy = nx.transitivity(G_noisy)/3
#     print('Global clustering coefficient error is:')
#     print(1-glo_cc_noisy/ground_glo_cc)
#
#     print('CC error is:')
#     cc_noisy = nx.average_clustering(G_noisy)
#     print(1-cc_noisy/ground_cc)
#
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

