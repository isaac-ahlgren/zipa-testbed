import os
import sys

import numpy as np
from sklearn.datasets import make_blobs

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
    signal[9::window_size] = 1

    chunks = IoTCupidProcessing.chunk_signal(signal, window_size)
    output = IoTCupidProcessing.compute_derivative_on_chunks(chunks)

    assert np.sum(output) == 0  # nosec
    assert len(output) == 100  # nosec


def test_detect_events():
    signal = np.zeros(100)
    signal[0:10] = 1
    signal[12:14] = 1
    signal[17:19] = 1
    signal[77] = 1
    signal[97:100] = 1

    events = IoTCupidProcessing.detect_event_on_chunks(signal, 0.1, 1, 20, 10)

    assert len(events) == 4  # nosec
    assert events[0][0] == 0 and events[0][1] == 140  # nosec
    assert events[1][0] == 170 and events[1][1] == 190  # nosec
    assert events[2][0] == 770 and events[2][1] == 780  # nosec
    assert events[3][0] == 970 and events[3][1] == 1000  # nosec


def test_get_event_signals():
    signal = [4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 4, 4, 5]
    events = IoTCupidProcessing.detect_event_on_chunks(signal, 4, 5, 1, 10)

    event_signals = IoTCupidProcessing.get_event_signals(events, signal, 10)

    assert len(event_signals) == 2  # nosec
    assert set(event_signals[0]) == set([4, 4, 4, 4])  # nosec
    assert set(event_signals[1]) == set([4, 4, 5])  # nosec


def test_group_events():
    signal = [4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 2, 3, 3, 4, 4, 5]
    events = IoTCupidProcessing.detect_event_on_chunks(signal, 4, 5, 1, 1)

    u = np.zeros((4, 3))
    u[0, 0] = 0.6
    u[0, 1] = 0.6
    u[0, 2] = 0

    u[1, 0] = 0
    u[1, 1] = 1
    u[1, 2] = 0.8

    u[2, 0] = 0.6
    u[2, 1] = 1
    u[2, 2] = 0.8

    u[3, 0] = 0
    u[3, 1] = 0
    u[3, 2] = 0

    grouped_events = IoTCupidProcessing.group_events(events, u, 0.5)

    assert len(grouped_events[3]) == 0  # nosec
    assert set(grouped_events[0]) == set([(0, 4), (7, 9)])  # nosec
    assert set(grouped_events[1]) == set([(7, 9), (13, 16)])  # nosec
    assert set(grouped_events[2]) == set([(0, 4), (7, 9), (13, 16)])  # nosec


def test_calculate_inter_event_timings():
    grouped_events = [
        [(1, 4), (500, 1000), (1300, 1500), (1800, 2100)],
        [(5, 7), (9, 16), (25, 27)],
        [(19, 21)],
    ]
    fps = IoTCupidProcessing.calculate_inter_event_timings(grouped_events, 1, 1, 8)

    IN_MICROSECONDS = 1000000

    assert_key1 = bytearray()
    assert_key1 += (499 * IN_MICROSECONDS).to_bytes(4, "big")
    assert_key1 += (800 * IN_MICROSECONDS).to_bytes(4, "big")

    assert_key2 = bytearray()
    assert_key2 += (4 * IN_MICROSECONDS).to_bytes(4, "big")
    assert_key2 += (16 * IN_MICROSECONDS).to_bytes(4, "big")

    assert len(fps) == 2  # nosec
    assert set(fps[0]) == set(assert_key1)  # nosec
    assert set(fps[1]) == set(assert_key2)  # nosec
