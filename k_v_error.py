import matplotlib.pyplot as mpl
import numpy as np

x = np.genfromtxt("sample_stocks.csv", delimiter = ',')

mpl.scatter(x[:,0], x[:,1], s=10)
mpl.show()

#different colors for different clusters
colors = 10*["r", "g", "b", "c", "k"]

class k_means:
    #initializing the k_means class ... like a constructor
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k=k;
        self.tol=tol
        self.max_iter=max_iter

    #function for fitting the data
    #making the clusters and cluster heads
    def fit(self, data):
        #dicrionary of centroids
        self.centroids = {}

        #initializing the centroids
        #to the first k data elements
        for i in range(self.k):
            self.centroids[i] = data[i]
        #print("Initial centroids: ", self.centroids)
        #carrying out the clustering process
        for i in range(self.max_iter):
            #dictionary of indices
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                #list of distances of the point from all the centroids
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                #finding the index of the minimimum distance
                classification = distances.index(min(distances))
                #appending the coordinates to the minimum index in classifications
                self.classifications[classification].append(featureset)

            #print("Classifications array in ", i, "th iteration: ", self.classifications)

            #dictionary of old centroids
            prev_centroids = dict(self.centroids)

            #finding new centroids
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            #checking the cluster heads for the tolerance
            for c in self.centroids:
                oldc = prev_centroids[c]
                newc = self.centroids[c]
                if np.sum((newc-oldc)/oldc*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

obj = k_means()
obj.fit(x)
