import os
import sys

import numpy as np

sys.path.insert(1, os.getcwd() + "/eval/perceptio")
sys.path.insert(1, os.getcwd() + "/eval")
sys.path.insert(1, os.getcwd() + "/src")

from perceptio_tools import extract_all_events
from signal_file import Signal_Buffer


def test_extract_all_events():
    buffer = np.zeros(2 * 10000 + 200)
    buffer[1:3] = 1
    buffer[9980:10020] = 1
    buffer[11500:11580] = 1
    buffer[20000:20020] = 1
    sb = Signal_Buffer(buffer)
    events = extract_all_events(sb, 2, 0.1, 20, 0.99)

    assert len(events) == 4  # nosec
    assert events[0][0] == 1 and events[0][1] == 3  # nosec
    assert events[1][0] == 9980 and events[1][1] == 10020  # nosec
    assert events[2][0] == 11500 and events[2][1] == 11580  # nosec
    assert events[3][0] == 20000 and events[3][1] == 20020  # nosec
