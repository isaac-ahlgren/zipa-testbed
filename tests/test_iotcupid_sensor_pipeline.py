import os
import sys

from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/src")

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402


def test_iotcupid_cmeans_clustering():

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.05, random_state=0)

    cntr, u, c, score = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        X.T, 5, 0.05, 1.5, 3, 100, 0.5
    )

    assert c == 3  # nosec
