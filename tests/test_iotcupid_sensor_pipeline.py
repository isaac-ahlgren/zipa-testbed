import os
import sys

from sklearn.datasets import make_blobs
import numpy as np

sys.path.insert(1, os.getcwd() + "/src")

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402


def test_iotcupid_cmeans_clustering():

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.05, random_state=0)

    cntr, u, c, score = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        X.T, 5, 0.05, 1.5, 3, 100, 0.5
    )

    assert c == 3  # nosec

def test_iotcupid_derivative():
    window_size = 10
    signal = np.zeros(1000)
    signal[::window_size] = 1

    output = IoTCupidProcessing.compute_derivative(signal, window_size)

    assert np.sum(output) == 0 # nosec
    assert len(output) == 990 # nosec
