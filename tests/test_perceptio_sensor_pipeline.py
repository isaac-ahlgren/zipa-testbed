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

    events = PerceptioProcessing.get_events(
        signal, 0.99, 0.1, 1.1, 2
    )

    assert len(events) == 4  # nosec
    assert events[0][0] == 0 and events[0][1] == 14  # nosec
    assert events[1][0] == 17 and events[1][1] == 19  # nosec
    assert events[2][0] == 77 and events[2][1] == 82  # nosec
    assert events[3][0] == 97 and events[3][1] == 100  # nosec

