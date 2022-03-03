import csv
import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


filename = 'data/traveldata.csv'


data = []
# with open(filename) as f:
#     reader = csv.reader(f)
#     header_row = next(reader)
#

csv_file = csv.reader(open(filename))
# print(csv_file)
for row in csv_file:
    data.append(row[1:])
    # print(row[1])

data = np.array(data[1:])
data = data.astype(float)

pred = KMeans(n_clusters=5, random_state=9).fit_predict(data)
print(pred)