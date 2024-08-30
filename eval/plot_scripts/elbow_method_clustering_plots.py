import os
import sys

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/../../src")

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402
from signal_processing.perceptio import PerceptioProcessing  # noqa: E402

CLUSTER_TH = 0.1
MAX_CLUSTER_SIZE = 5


def perceptio_wrapper(X, clusters):
    labels, k, inertias = PerceptioProcessing.kmeans_w_elbow_method(
        X, clusters, CLUSTER_TH
    )
    return inertias


def iotcupid_wrapper(X, clusters):
    cntr, u, c, scores = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        X.T, clusters, CLUSTER_TH, 1.5, 3, 100, 0.5
    )
    return scores


def clustering_data(wrapper, cluster_size, max_cluster_size):
    X, _ = make_blobs(
        n_samples=300, centers=cluster_size, cluster_std=0.05, random_state=0
    )
    scores = wrapper(X, max_cluster_size)
    return scores


def plot_cluster_scores(scores, saveplot=True, filename=None):
    plt.plot(scores)
    plt.xlabel("Cluster Size Used")
    plt.ylabel("Score")
    if saveplot:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()


def main():
    cluster1_scores_iot = clustering_data(iotcupid_wrapper, 1, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster1_scores_iot, filename="./plot_data/iotcupid_cluster1.png"
    )

    cluster2_scores_iot = clustering_data(iotcupid_wrapper, 2, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster2_scores_iot, filename="./plot_data/iotcupid_cluster2.png"
    )

    cluster3_scores_iot = clustering_data(iotcupid_wrapper, 3, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster3_scores_iot, filename="./plot_data/iotcupid_cluster3.png"
    )

    cluster1_scores_perc = clustering_data(perceptio_wrapper, 1, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster1_scores_perc, filename="./plot_data/perceptio_cluster1.png"
    )

    cluster2_scores_perc = clustering_data(perceptio_wrapper, 2, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster2_scores_perc, filename="./plot_data/perceptio_cluster2.png"
    )

    cluster3_scores_perc = clustering_data(perceptio_wrapper, 3, MAX_CLUSTER_SIZE)
    plot_cluster_scores(
        cluster3_scores_perc, filename="./plot_data/perceptio_cluster3.png"
    )


if __name__ == "__main__":
    main()
