import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402

goldsig_rng = np.random.default_rng(0)


def golden_signal(sample_num):
    return goldsig_rng.integers(0, 10, size=sample_num)


adv_rng = np.random.default_rng(12345)


def adversary_signal(sample_num):
    return adv_rng.integers(0, 10, size=sample_num)


def gen_min_events(
    signal,
    chunk_size,
    min_events,
    top_th,
    bottom_th,
    agg_th,
    a,
    window_size,
):
    events = []
    event_signals = []
    iteration = 0
    while len(events) < min_events:
        chunk = signal.read(chunk_size)

        smoothed_data = IoTCupidProcessing.ewma(chunk, a)

        derivatives = IoTCupidProcessing.compute_derivative(smoothed_data, window_size)

        received_events = IoTCupidProcessing.detect_events(
            abs(derivatives), bottom_th, top_th, agg_th
        )

        if len(received_events) != 0:
            received_event_signals = IoTCupidProcessing.get_event_signals(
                received_events, smoothed_data
            )

            for i in range(len(received_events)):
                received_events[i] = (
                    received_events[i][0] + chunk_size * iteration,
                    received_events[i][1] + chunk_size * iteration,
                )

            # Reconciling lumping adjacent events across windows
            if (
                len(received_events) != 0
                and len(events) != 0
                and received_events[0][0] - events[-1][1] <= agg_th
            ):
                events[-1] = (events[-1][0], received_events[0][1])
                event_signals[-1] = np.append(
                    event_signals[-1][1], received_event_signals
                )

                events.extend(received_events[1:])
                event_signals.extend(received_event_signals[1:])
            else:
                events.extend(received_events)
                event_signals.extend(received_event_signals)
        iteration += 1
    return events, event_signals


def generate_bits(
    events,
    event_signals,
    max_clusters,
    cluster_th,
    m_start,
    m_end,
    m_searches,
    quantization_factor,
    feature_dim,
    Fs,
    key_size_in_bytes,
    mem_th,
):
    event_features = IoTCupidProcessing.get_event_features(event_signals, feature_dim)

    cntr, u, optimal_clusters, fpcs = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        event_features.T, max_clusters, cluster_th, m_start, m_end, m_searches, mem_th
    )

    grouped_events = IoTCupidProcessing.group_events(events, u, mem_th)

    inter_event_timings = IoTCupidProcessing.calculate_inter_event_timings(
        grouped_events, Fs, quantization_factor, key_size_in_bytes
    )

    return inter_event_timings, grouped_events
