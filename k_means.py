import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Kmeans:
    def __init__(self,k=3,max_iterations=1000):
        self.k = k
        self.max_iterations = max_iterations

    def euclidean_dist(self,X,centroids):
        # print(X.shape)
        # print(centroids.shape)
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis,:,:], axis=2)
        return dists
    
    def fit(self,X):
        X = X.to_numpy()
        self.X = X    
        self.n = X.shape[0]
        self.feature_size = X.shape[1]
        # print(self.feature_size)
        # self.centroids = X.sample(n=self.k).to_numpy()
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        # print(self.centroids.shape)
        for a in range(self.max_iterations):
            distances = self.euclidean_dist(X, self.centroids)  # Calculate distances to centroids
            X_clusters = np.argmin(distances, axis=1)  # Assign each point to the nearest centroid
            new_centroids = np.empty((self.k,self.feature_size))
            for i in range(self.k):
                indices_in_clus = np.where(X_clusters == i)[0]  # Get indices of points in the current cluster
                if len(indices_in_clus) == 0:
                    new_centroids[i] = self.centroids[i]  # If no points are assigned, keep the old centroid
                else:
                    cluster_points = X[indices_in_clus].reshape(-1, self.feature_size)
                    new_centroids[i] = np.mean(cluster_points, axis=0) # now update centroids by the mean value of their cluster

            if np.max(np.abs((self.centroids - np.array(new_centroids)))) < 0.0001:   # If all centroids are still the same as the old ones means the data has converged
                print(f'Convergence reached after {a} iterations')
                break

            self.centroids = new_centroids
        return self.centroids

    def predict(self,X):
        distances = self.euclidean_dist(X.values,self.centroids)
        X_clusters = np.argmin(distances,axis=1)
        return X_clusters
    
    def getCost(self,X,X_clusters):
        wcss = 0
        for i in range(self.k):
            cluster_i = X.values[X_clusters == i]
            centroid_i = self.centroids[i]
            squared_distances = np.sum((cluster_i - centroid_i) ** 2, axis=1)
            wcss += np.sum(squared_distances)
        return wcss

    def findElbowPoint(self,X,endK):
        wcss = []
        for k in range(1,endK+1):
            self.k = k
            self.fit(X)
            X_clusters = self.predict(X)
            wcss.append(self.getCost(X,X_clusters))

        plt.figure(figsize=(8, 6))  # used chatgpt for plotting
        plt.plot(range(1, endK + 1), wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.xticks(range(1, endK + 1))
        plt.grid(True)
        # plt.savefig('/Users/masumi/Desktop/Study/Sem5/SMAI/Assignments/A2/smai-m24-assignments-MasumiD/assignments/2/figures/elbow_plot_reduced_dataset.png')
        plt.show()

        return wcss
