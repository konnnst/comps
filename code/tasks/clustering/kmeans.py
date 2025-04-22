import numpy as np
import matplotlib.pyplot as plt

from .utils import generate_dataset


class KMeans:
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def fit(self, points):
        n_samples = points.shape[0]

        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = points[random_indices]

        for _ in range(self.max_iterations):
            # Assign each point to the nearest centroid
            distances = np.sqrt(((points - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # Calculate new centroids
            new_centroids = np.array([points[self.labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            centroid_shift = np.sqrt(((new_centroids - self.centroids)**2).sum(axis=1)).max()
            if centroid_shift < self.tolerance:
                break

            self.centroids = new_centroids

        return self

    def predict(self, points):
        distances = np.sqrt(((points - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def visualize(self, points, title="K-Means clustering"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.k):
            cluster_points = points[self.labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                       label=f'Cluster {i+1}', alpha=0.7)

        # Plot centroids
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2],
                   marker='*', s=300, c='black', label='Centroids')

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    dataset = generate_dataset(6, range(5, 20), 10, 100)
    kmeans = KMeans(k=3)
    kmeans.fit(dataset)

    kmeans.visualize(dataset)

    new_points = np.array([[0, 0, 0], [5, 5, 5], [-5, 5, 0], [2, 2, 2]])
    predicted_labels = kmeans.predict(new_points)
    print("Predicted cluster labels for new points:", predicted_labels)
