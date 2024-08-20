import sys
import os
from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/../../src")

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402
from signal_processing.perceptio import PerceptioProcessing # noqa: E402

CLUSTER_TH = 0.1

def perceptio_wrapper(X, clusters):
    labels, k, inertias = PerceptioProcessing.kmeans_w_elbow_method(X, clusters, CLUSTER_TH)
    return inertias

def iotcupid_wrapper(X, clusters):
    cntr, u, c, scores = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        X.T, clusters, 0.05, 1.5, 3, 100, 0.5
    )
    return scores

def clustering_data(wrapper, ):
    X, _ = make_blobs(n_samples=300, centers=cluster, cluster_std=0.05, random_state=0)
    labels, k, inertias = PerceptioProcessing.kmeans_w_elbow_method(X, )
    return inertias

def main():
    
