import numpy as np

x = np.genfromtxt("coordinates.csv", delimiter = ',')
print(x)
centroids={}
for i in range(0,3):
    centroids[i] = []
print(centroids)
