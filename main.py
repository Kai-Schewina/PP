from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pandas as pd

n_centroids = 3
n_features = 3


def make_data(n_samples=100, n_centroids=3, n_features=2):
    features, true_labels = make_blobs(n_samples=n_samples, centers=n_centroids, n_features=n_features, random_state=42)
    scaled_features = StandardScaler().fit_transform(features)

    min_val = scaled_features.min()
    max_val = scaled_features.max()

    return scaled_features, min_val, max_val


data, lower, upper = make_data(n_centroids=n_centroids, n_features=n_features)


def initialize_centroids(lower, upper, n_centroids, n_features):
    centroids = np.random.uniform(lower, upper, [n_centroids, n_features])
    #for i in range(0, n_centroids):
    #    centroids.append([np.random.uniform(lower, upper)], [np.random.uniform(lower, upper)])
    return centroids


centroids = initialize_centroids(lower, upper, n_centroids, n_features)


def assign_clusters(data, centroids):
    clusters = []
    for e in data:
        dist_list = []
        for i, c in enumerate(centroids):
            dist = [(a - b) ** 2 for a, b in zip(e, c)]
            dist = math.sqrt(sum(dist))
            dist_list.append(dist)
        clusters.append(dist_list.index(min(dist_list)))
    return clusters


clusters = assign_clusters(data, centroids)


def reposition_centroids(data, centroids, n_centroids):
    for i in range(0, n_centroids):
        cluster = [x for x in data if x[0] == n_centroids]
        



