import matplotlib.pyplot as mpl
import numpy as np

x = np.genfromtxt("coordinates.csv", delimiter = ',')
print(x)

mpl.scatter(x[:,0], x[:,1], s=150)
mpl.show()

#different colors for different clusters
colors = 10*["r", "g", "b", "c", "k"]

class fuzzy_c:
    def __init__(self, max_iter=300, tol=0.001, k=3):
        self.max_iter=max_iter
        self.tol=tol
        self.k=k

    def fit(self, data):
        self.cent = {}
        for i in range(self.k):
            self.cent[i]=data[i]

        for i in range(self.max_iter):
            self.cluster = {}
            for j in range(self.k):
                self.cluster[j] = []

            self.u = [len(data)]
            for j in range(len(data)):
                self.u[j] = []
            print(self.u)

            for d in data:
                distances = [np.linalg.norm(d-self.cent[c]) for c in self.cent]
                sumDist = sum(distances)
                self.u[d] = [ 1/(distances[dist]/sumDist) for dist in distances]

            print("u:")
            print(self.u)

            for d in data:
                classification = self.u.index(min(self.u[:,j]))
                self.cluster[classification].append(d)

            print("cluster:")
            print(self.cluster)

            pcent = dict(self.cent)

            for i in range(self.k):
                self.cent[i] = sum([self.u[i][j]*self.u[i][j]*data[i] for j in self.u])/sum([self.u[i][j]*self.u[i][j] for j in self.u])

            optimized = True

            for c in self.centroids:
                oldc = pcent[c]
                newc = self.cent[c]
                if np.sum((newc-oldc)/oldc*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

            mpl.ion()
            mpl.cla()
            for c in self.cent:
                mpl.scatter(self.cent[c][0], self.cent[c][1], marker="o", color="k", s=150, linewidth=5)

            for classification in self.cluster:
                color = colors[classification]
                for featureset in self.cluster[classification]:
                    mpl.scatter(featureset[0], featureset[1], marker="o", color=color, s=10)

            mpl.show()
            mpl.pause(0.01)

obj = fuzzy_c()
obj.fit(x)
