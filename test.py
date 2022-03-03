import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.axisartist as axisartist
from scipy.stats import entropy
data_path = "cancer.txt"



def add_laplace_noise(data_list,epsilon):
    laplace_noise = np.random.laplace(0, 1/epsilon, len(data_list)) # 为原始数据添加μ为0，b为1的噪声
    # noise_list=laplace_noise + data_list
    return laplace_noise + data_list


a = [1, 2, 3]
b = random.choices(a, k=4)
c = random.sample(a, 4)
print(b)
print(c)