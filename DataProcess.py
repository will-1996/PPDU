import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, perm
# import pandas as pd


# facebook=4039     enron=36692        astro=18772          gplus=107614

Graph_size = 36692
File_length = 367662
miu = 200
m0 = 50
Group_length = Graph_size // m0
threshold = 0.4
threshold_wei = 0.2


def SubGraphGen(Graph_size, File_length, read_path, write_path):
    Graph = np.zeros((Graph_size, Graph_size))
    f = open(read_path, "r")  # 设置文件对象
    for i in range(File_length):  # 直到读取完文件
        line = f.readline()[:-1].strip().split(' ')  # 读取一行文件，去掉换行符
        Graph[int(line[0]) - 1][int(line[1]) - 1] = int(1)
        # Graph[int(line[1])][int(line[0])] = int(1)
    f.close()  # 关闭文件
    # 存储图数据
    np.savetxt(write_path, Graph, fmt='%d', delimiter=',')
    return


def degree_compute(graph):
    return np.sum(graph, axis=1)


def Add_noise(degree_vector, delta_f, epsilon):
    """
    :param degree_vector: 度向量
    :param delta_f: 敏感度
    :param epsilon: 隐私预算
    """
    noised_degree_vector = degree_vector + np.random.laplace(loc=0, scale=(delta_f / epsilon), size=degree_vector.shape)
    return noised_degree_vector


def Degree_distri(degree, noi_deg=0):
    degree = np.sort(degree)
    noi_deg = np.sort(noi_deg)
    plt.plot(degree)
    plt.plot(noi_deg)
    plt.show()


def degree_vector_compute(total_num, cluster_num, graph):
    """
    计算度向量：
    将邻接向量以列为单位 随机排序
    循环求和，逐组遍历，生成1w*100的数组，代表度向量
    """
    graph = graph.T
    np.random.shuffle(graph)
    graph = graph.T
    degree_vector = np.zeros([total_num, cluster_num])
    group_len = total_num // cluster_num + 1
    for i in range(total_num):
        for j in range(cluster_num - 1):
            degree_vector[i][j] = np.sum(graph[i][j * group_len:(j + 1) * group_len])
        degree_vector[i][-1] = np.sum(graph[i][(cluster_num - 1) * group_len:])
    return degree_vector


def degree_vector_com2(graph_size, m0, graph, degree):
    """

    :param graph_size:
    :param m0:
    :param graph:
    :param degree:
    :return:
    """
    graph = graph[np.argsort(degree)]

    degree_vector = np.zeros([graph_size, m0])
    for i in range(graph_size):
        for j in range(m0):
            degree_vector[i][j] = np.sum(graph[i][j:-1:m0])
    return degree_vector


def similarity_com(vector1, vector2, degree1, degree2):
    common_neig = np.minimum(vector1, vector2)
    com = np.sum(common_neig)
    return com / np.sqrt(degree1 * degree2)


def SCAN(degree_vector):
    """

    :param degree_vector: 根据度值按降序排列
    :return:
    """
    degree = np.sum(degree_vector, axis=1)
    degree_vector = degree_vector[np.argsort(degree)][::-1]
    degree = np.sort(degree)[::-1]

    for i in range(len(degree)):
        if degree[i] == 0:
            degree[i] = 1

    sim_matrix = np.zeros([len(degree_vector), len(degree_vector)])
    for i in range(len(degree_vector)):
        # 仅对右上 三角 赋值
        for j in range(i + 1, len(degree_vector)):
            sim_matrix[i][j] = similarity_com(degree_vector[i], degree_vector[j], degree[i], degree[j])

    return sim_matrix
    # pass


def Cluster_SCAN(sim_matrix, degree, thr=threshold):
    # 补全相似度矩阵
    for ii in range(len(sim_matrix)):
        for jj in range(0, ii):
            sim_matrix[ii][jj] = sim_matrix[jj][ii]

    ind = np.argsort(degree)[::-1]
    results = []
    i = 0
    while np.sum(sim_matrix) != 0:
        results.append([])  # 存放结果ID
        temp = []  # 存放执行ID

        # 选取当前相似度矩阵中度最大的节点ID，放入列表
        degree_temp = np.sum(sim_matrix, axis=1)

        for j in range(len(sim_matrix)):
            # print('j is %d', j)
            if degree_temp[j] != 0:
                temp.append(j)
                results[i].append(ind[j])
                # 将对应节点的相似度向量置0
                # sim_matrix[j] = 0
                # sim_matrix[:, j] = 0
                break
        while temp:

            # s是当前节点相似度按执行ID的倒序数组
            s = np.argsort(sim_matrix[temp[0]])[::-1]

            # print(s[:4])
            # print(sim_matrix[temp[0]])
            for k in range(miu):
                # s[k]是和当前节点 第k相似的节点的执行ID
                if sim_matrix[temp[0]][s[k]] >= thr:

                    temp.append(s[k])
                    results[i].append(ind[s[k]])
                    sim_matrix[:, s[k]] = 0
                else:
                    break
            sim_matrix[temp[0]] = 0
            sim_matrix[:, temp[0]] = 0
            temp.pop(0)

        i += 1

    return results


def Weight(a, b, c):
    m = np.minimum(b, c)
    ww = []
    for i in range(len(b)):
        w = 0
        for j in range(int(m[i]) + 1):
            w += comb(b[i], j) * comb((a - b[i]), (c[i] - j)) * j / comb(a, c[i])
        ww.append(w)
    return np.array(ww)


def weight_simi_com(len, vector1, vector2, degree1, degree2):
    w = Weight(len, vector1, vector2)
    com = np.sum(w)
    return com / np.sqrt(degree1 * degree2)


def Wei_SCAN(degree_vector):
    degree = np.sum(degree_vector, axis=1)
    degree_vector = degree_vector[np.argsort(degree)][::-1]
    degree = np.sort(degree)[::-1]
    sim_matrix = np.zeros([len(degree_vector), len(degree_vector)])
    for i in range(len(degree_vector)):
        # 仅对右上 三角 赋值
        for j in range(i + 1, len(degree_vector)):
            sim_matrix[i][j] = weight_simi_com(Group_length, degree_vector[i], degree_vector[j], degree[i], degree[j])
    return sim_matrix


# SubGraphGen(Graph_size, File_length, 'data/Email-Enron.txt', 'data/Enron/graph.txt')

Graph = np.loadtxt('data/GraphEnron', dtype=int, delimiter=',')
print(Graph.shape)

Degree = degree_compute(Graph)
np.savetxt('data/degreeEnron.txt', Degree, fmt='%d', delimiter=',')

# Degree_sorted = np.sort(Degree)

# Noisy_degree = Add_noise(Degree, 1, 0.5)
# # print(max(Degree-Noisy_degree))
# # Degree_distri(Degree, Noisy_degree)
#
# degree_vector = degree_vector_com2(Graph_size, m0, Graph, Degree)[::-1]  # 已排序的度向量
#
# # degree_vector = degree_vector_compute(Graph_size, 10, Graph)
#
#
# # print(np.abs(np.sum(degree_vector, axis=1)-np.sort(Degree)))
#
# # noisy_deg_vec = Add_noise(degree_vector, 1, 1)
# # print(np.sum(np.abs(degree_vector[:10]-noisy_deg_vec[10]))/10)
#
# # simlarity_matrix = SCAN(degree_vector[:Graph_size//2])
#
# Graph = Graph[np.argsort(Degree)][::-1]
#
# simlarity_matrix_graph = SCAN(Graph)
#
# resu = Cluster_SCAN(simlarity_matrix_graph, Degree)
# # res = Cluster_SCAN(simlarity_matrix, Degree, 0.6)
# for i in range(len(resu)):
#     print('第%d组为:' % (i + 1), resu[i])
#
#     # print(res[i])
#     # print(resu[i])
# # print(resu)
#
# # print(simlarity_matrix[0]-simlarity_matrix_graph[0])
# # print(np.sum(np.abs(simlarity_matrix[0]-simlarity_matrix_graph[0])))
# # print(simlarity_matrix_graph[0])
#
# # to do: 构建m0与点数阈值μ和相似度错误比例之间的关系，   μ：前μ个近似的节点
#
#
# # print(simlarity_matrix[0])


# for i in range(10):
#     print(np.sort(simlarity_matrix)[test+30*i][-5:])
#     print(np.sort(simlarity_matrix_graph)[test+30*i][-5:])
#     print('##############################################################')
# for i in range(10):
#     print(np.argsort(simlarity_matrix[test+i])[::-1][:5])
#     print(np.argsort(simlarity_matrix_graph[test+i])[::-1][:5])
#     print('##############################################################')


# print(np.argsort(simlarity_matrix[10])[-miu:])
# print(np.argsort(simlarity_matrix_graph[10])[-miu:])

