import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from .utils import generate_dataset


class Hierarchical:
    def __init__(self, k=3):
        self.k = k
        self.labels = None
        self.linkage_matrix = None

    def fit(self, points):
        distances = pdist(points)
        self.linkage_matrix = linkage(distances, method="ward")
        self.labels = self._cut_dendrogram(self.linkage_matrix, self.k)

        return self

    def _cut_dendrogram(self, linkage_matrix, k):
        return fcluster(linkage_matrix, k, criterion="maxclust") - 1

    def plot_dendrogram(self, title="Hierarchical Clustering Dendrogram"):
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title(title)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()

    def visualize(self, points, title="Hierarchical clustering"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(self.k):
            cluster_points = points[self.labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                       label=f"Cluster {i+1}", alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()



if __name__ == "__main__":
    dataset = generate_dataset(6, range(5, 20), 10, 100)

    hc = Hierarchical(k=3)
    hc.fit(dataset)

    hc.visualize(dataset)
    hc.plot_dendrogram()
