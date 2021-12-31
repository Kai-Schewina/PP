from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pandas as pd
from line_profiler import LineProfiler
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from itertools import chain
import time


def make_data(n_samples=100, n_centroids=3, n_features=2):
    features, true_labels = make_blobs(n_samples=n_samples, centers=n_centroids,
                                       n_features=n_features, random_state=42)

    scaled_features = StandardScaler().fit_transform(features)

    min_val = scaled_features.min()
    max_val = scaled_features.max()
    scaled_features = pd.DataFrame(scaled_features)

    return scaled_features, min_val, max_val


def initialize_centroids(lower, upper, n_centroids, n_features):
    np.random.seed(42)
    centroids = np.random.uniform(lower, upper, [n_centroids, n_features])
    return centroids


def assign_clusters(chunk, centroids):
    clusters = []
    for index, e in chunk.iterrows():
        dist_list = []
        for i, c in enumerate(centroids):
            dist = [(a - b) ** 2 for a, b in zip(e, c)]
            dist = math.sqrt(sum(dist))
            dist_list.append(dist)
        clusters.append(dist_list.index(min(dist_list)))
    return clusters


def reposition_centroids(data, centroids, n_centroids, n_features):
    new_centroids = centroids
    for i in range(n_centroids):
        for j in range(n_features):
            new_centroids[i, j] = data.loc[data['cluster'] == i, [j]].sum() / data[data['cluster'] == i].shape[0]
    return new_centroids


def check_stop(data, new_clusters):
    new_clusters = pd.DataFrame(new_clusters, columns=['cluster'])
    if data["cluster"].equals(new_clusters["cluster"]):
        return True
    else:
        return False


def main():
    start_time = time.time()

    n_centroids = 5
    n_features = 6
    n_samples = 1000000

    data, lower, upper = make_data(n_samples=n_samples, n_centroids=n_centroids, n_features=n_features)
    centroids = initialize_centroids(lower, upper, n_centroids, n_features)

    # Chunking the data
    num_processes = mp.cpu_count()
    chunk_size = int(data.shape[0]/num_processes)
    chunks = [data.iloc[data.index[i:i + chunk_size]] for i in range(0, data.shape[0], chunk_size)]

    pool = mp.Pool(processes=num_processes)

    # Initial Clustering
    start_time = time.time()
    result = pool.map(partial(assign_clusters, centroids=centroids), chunks)
    clusters = list(chain.from_iterable(result))
    end_time = time.time()
    print(f'Completed in {end_time - start_time:.2f} secs.')
    data["cluster"] = clusters

    counter = 0
    while True:
        centroids = reposition_centroids(data, centroids, n_centroids, n_features)

        start_time = time.time()
        result = pool.map(partial(assign_clusters, centroids=centroids), chunks)
        clusters = list(chain.from_iterable(result))
        end_time = time.time()
        print(f'Completed in {end_time - start_time:.2f} secs.')

        stop = check_stop(data, clusters)
        if stop:
            print("Algorithm converged after " + str(counter) + " iterations.")
            break
        else:
            data["cluster"] = clusters
            counter += 1

    end_time = time.time()
    print(f'Completed in {end_time - start_time:.2f} secs.')


if __name__ == '__main__':
    main()
