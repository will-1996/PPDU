import numpy as np
import matplotlib.pyplot as plt

"""实验结果， facebook dk序列总和为 176468
随着epsilon增加，LDP和DP的差异倍数也会增加
ep=0.5 时， 倍率约为 3.2
ep=8 时，   倍率约为18.57
可见LDP方法对噪音不太敏感
"""


ep = 8

def get_seq(read_path):
    f = open(read_path, 'r')
    seq = []
    for line in f:
        a = line.strip().split(',')
        a = [int(i) for i in a]
        # map(int, a)
        seq.append(a)
    return seq


def searchDK2(seq, degree):
    matr = np.zeros([1056, 1056]).astype(int)
    for i in range(len(seq)):
        for j in seq[i]:
            ind1 = degree[i].astype(int)
            ind2 = degree[j].astype(int)
            matr[ind1, ind2] += 1
    return matr


def showDK2(matr):
    for i in range(len(matr)):
        for j in range(len(matr[i])):
            if matr[i, j]:
                print(i, j, matr[i, j])


def Add_noise(degree_vector, delta_f, epsilon):
    """
    :param degree_vector: 度向量
    :param delta_f: 敏感度
    :param epsilon: 隐私预算
    """
    noised_degree_vector = degree_vector + np.random.laplace(loc=0, scale=(delta_f / epsilon), size=degree_vector.shape)
    noised_degree_vector = np.round(noised_degree_vector)
    for i in range(len(noised_degree_vector)):
        for j in range(len(noised_degree_vector[i])):
            if noised_degree_vector[i, j] < 0:
                noised_degree_vector[i, j] = 0
    return noised_degree_vector


def getUserDK(Seq, noisyDegree):
    """
    生成user端的DK2 序列，
    :param Seq:
    :param noisyDegree:
    :return:
    """
    dk = []
    for i in range(len(Seq)):
        dk.append([])
        for k in range(1056):
            dk[i].append(0.0)
        # [dk[i].append([0]) for k in range(1046)]

        for j in Seq[i]:
            ind = noisyDegree[j].astype(int)
            dk[i][ind] += 1
    return dk


def dkPer(dk, epsilon, noisyDeg):
    """
    对用户端的dk序列进行分别扰动
    :param dk:
    :param epsilon:
    :param noisyDeg:
    :return: 返回的dk的行索引代表的是结点的编号，而非dk中的第一个index，通过noisyDegree转换为newDK
    """
    newDK = np.zeros([1056, 1056])
    for i in range(len(dk)):
        for j in range(len(dk[i])):
            if dk[i][j] > 0:
                dk[i][j] += np.random.laplace(loc=0, scale=(1/ep))

                dk[i][j] = np.round(dk[i][j]).astype(int)
                if dk[i][j] < 0:
                    dk[i][j] = 0
    for i in range(len(dk)):
        ind1 = noisyDeg[i].astype(int)
        newDK[ind1] += dk[i]
    return newDK

# Graph = np.loadtxt('data/Facebook_graph', delimiter=',')

Degree = np.loadtxt('data/facebook_degree', delimiter=',')

Seq = get_seq('data/facebook_seq')

matr = searchDK2(Seq, Degree)       # matr是缺省值为0的DK2矩阵，索引代表两个度值，取值代表DK的数量

nonZero = []                        # 未加噪前 dk取值不为0的索引
for i in range(len(matr)):
    for j in range(len(matr[i])):
        if matr[i, j]:
            nonZero.append([i, j])
nonZero = np.array(nonZero)

noisyDK2 = Add_noise(matr, 1, ep)

differ = noisyDK2-matr

temp1 = 0
for i in nonZero:
    temp1 += np.abs(differ[i[0], i[1]])

    # print(differ[i[0], i[1]])


noisyDegree = np.round(Degree+np.random.laplace(loc=0, scale=(1 / ep), size=Degree.shape))
for i in range(len(noisyDegree)):
    if noisyDegree[i] < 0:
        noisyDegree[i] = 0

noisyMatr = searchDK2(Seq, noisyDegree)     # 基于公开的加噪度值计算出的DK序列
# 真实情况应该是user各自扰动自身持有的DK数据，所以应建立用户长度的列表


dk = getUserDK(Seq, noisyDegree)
dk = np.array(dk)
distriDK2 = dkPer(dk, ep, noisyDegree)


print(np.sum(matr))
# distriDK2 = Add_noise(noisyMatr, 1, 1)      # 对fake序列加噪
print(temp1)
print(np.sum(np.abs(distriDK2-matr))/temp1)
# print(np.sum(np.abs(noisyMatr-matr))/temp1)
