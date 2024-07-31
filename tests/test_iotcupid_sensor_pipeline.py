import os
import sys
from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/src")
from signal_processing.iotcupid import IoTCupidProcessing


def test_iotcupid_cmeans_clustering():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
    
    cntr, u, c, fpc = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(X.T, 5, 0.1, 1.1, 3, 10)
    
    assert c == 3