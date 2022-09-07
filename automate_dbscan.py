# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:56:56 2022

@author: MIKLOS
"""

import os
import pandas as pd
import numpy as np
import numpy.matlib

from sklearn.cluster import DBSCAN  # for building a clustering model
from sklearn.preprocessing import MinMaxScaler  # for feature scaling
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs  # for demo purpases
import matplotlib.pyplot as plt


# Adjust font family and font size
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 18})
plt.rcParams["figure.figsize"] = (16, 12)


def get_knee_point(curve):
    """Gets the index of the knee point.
    Idea: draw a line from the first to the last point of the curve and then
    find the data point that is farthest away from that line.
    """
    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T
    np.array([range(n_points), curve])
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - first_point
    scalar_product = np.sum(
        vec_from_first * np.matlib.repmat(line_vec_norm, n_points, 1), axis=1
    )
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    idx_of_best_point = np.argmax(dist_to_line)

    return idx_of_best_point


def auto_dbscan(df):
    """
    This funtion first determines the optimal value for epsilon in a DBSCAN
    cluster algorithm using nearest neighbor approach with reading the knee
    point, then performing an DBSCAN clustering to determine the clusters in
    the data with the optimal epsilon value.

    Parameters
    ----------
    df : DataFrame
        This is the input DataFrame matrix where the data is ideally scalled.

    Returns
    -------
    df : DataFrame
        The last column of this DataFrame is the column with the clusters.

    """

    if not os.path.exists(PATH_DBSCAN_KNEE):
        os.makedirs(PATH_DBSCAN_KNEE)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, _ = nbrs.kneighbors(df)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    best_index = get_knee_point(distances)
    epsilon = distances[best_index]

    plt.plot(distances)
    plt.title("K-distance Graph for the optimal epsilon")
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel("Epsilon")
    plt.axhline(epsilon, color='r')
    plt.axvline(best_index, color="y")

    name = "DBSCAN_epsilon_knee_plot.png"
    plt.savefig(f"{PATH_DBSCAN_KNEE}{name}")

    model_dbscan = DBSCAN(eps=epsilon)

    yhat = model_dbscan.fit_predict(df)
    df["clusters_DBSCAN"] = yhat
    num_of_clusters = len(df["clusters_DBSCAN"].unique())

    print(f"Optimal value of epsilon: {epsilon}")
    print(f"Number of clusters in data based on DBSCAN: {num_of_clusters}")

    return df


PATH_DATA = "./Data/"
PATH_DBSCAN_KNEE = "./Dbscan_knee/"


if __name__ == "__main__":

    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                      random_state=0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)

    X_scaled = auto_dbscan(X_scaled)
