import numpy as np

from dpm import DPM
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


num_clusters = 64
num_dims = 10
num_points = 100000

data, _ = make_blobs(
    num_points, num_dims, centers=num_clusters, center_box=(-100, 100), random_state=42
)

print("Starting DPM clustering...")

dpm = DPM(
    data=data,
    bounds=(-100, 100),
    epsilon=1.0,
    delta=1 / (np.sqrt(num_points) * num_points),
)

_, clusters = dpm.perform_clustering()

print("Evaluating...")

labels = np.zeros(num_points)
for i, cluster in enumerate(clusters):
    labels[cluster] = i

silh_score = silhouette_score(data, labels, sample_size=10000)
print(f"Silhouette score: {silh_score}")
