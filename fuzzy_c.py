import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
import operator
import math
import csv
import pandas as pd

#X = pd.read_csv('example.csv')

#X = np.array(X1)
#print(X)

X = np.array([[3,3],[4,10],[9,6],[14,8],[18,11],[21,7]])
print(X)
k=2
MAX_ITER = 100
m = 2
df = 2

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()


colors = 10*["g","r","c","b","k"]


def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(len(X)):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def calculateClusterCenter(membership_mat):
    cluster_mem_val_1 = list(zip(*membership_mat))
    cluster_mem_val=[]
    for i in cluster_mem_val_1:
        cluster_mem_val.append(list(i))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(len(X)):
            data_point = X[i]
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(len(X)):
        x = X[i]
        distances = [np.linalg.norm(list(map(operator.sub, x,cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(len(X)):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    print("mem mat: ", membership_mat)
    return cluster_labels, cluster_centers



labels, centers = fuzzyCMeansClustering()
print("labels and centers: ", labels,centers)
