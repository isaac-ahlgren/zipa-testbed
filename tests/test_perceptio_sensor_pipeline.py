import os
import sys

import numpy as np
from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/src")

from signal_processing.perceptio import PerceptioProcessing  # noqa: E402


def test_detect_events():
    signal = np.zeros(100)
    signal[0:10] = 1
    signal[12:14] = 1
    signal[17:19] = 1
    signal[77] = 1
    signal[79] = 1
    signal[81] = 1
    signal[97:100] = 1

    events = PerceptioProcessing.get_events(signal, 0.99, 0.1, 1.1, 2)

    assert len(events) == 4  # nosec
    assert events[0][0] == 0 and events[0][1] == 14  # nosec
    assert events[1][0] == 17 and events[1][1] == 19  # nosec
    assert events[2][0] == 77 and events[2][1] == 82  # nosec
    assert events[3][0] == 97 and events[3][1] == 100  # nosec


def test_ewma():
    def test_func(data, alpha):
        ewma_data = np.zeros(len(data))
        ewma_data[0] = data[0]
        for i in range(1, len(ewma_data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        return ewma_data

    signal = np.zeros(100, dtype=np.int64)
    signal[0:25] = 1
    signal[27:88] = 2
    filtered1 = PerceptioProcessing.ewma(signal, 0.5)
    filtered2 = test_func(signal, 0.5)

    assert np.array_equal(filtered1, filtered2)  # nosec
