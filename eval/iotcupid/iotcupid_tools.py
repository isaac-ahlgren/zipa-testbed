import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src

from protocols.iotcupid import IoTCupid_Protocol  # noqa: E402

goldsig_rng = np.random.default_rng(0)


def golden_signal(sample_num):
    return goldsig_rng.integers(0, 10, size=sample_num)


adv_rng = np.random.default_rng(12345)


def adversary_signal(sample_num):
    return adv_rng.integers(0, 10, size=sample_num)


def get_events(arr, top_th, bottom_th, agg_th, a, window_size, feature_dim):
    smoothed_data = IoTCupid_Protocol.ewma_filter(arr, a)
    derivatives = IoTCupid_Protocol.compute_derivative(smoothed_data, window_size)
    events = IoTCupid_Protocol.detect_events(derivatives, bottom_th, top_th, agg_th)
    event_features = IoTCupid_Protocol.get_event_features(events, arr, feature_dim)

    return events, event_features


def gen_min_events(
    signal,
    chunk_size,
    min_events,
    top_th,
    bottom_th,
    agg_th,
    a,
    window_size,
    feature_dim
):
    events = []
    event_features = []
    iteration = 0
    while len(events) < min_events:
        chunk = signal.read(chunk_size)

        smoothed_data = IoTCupid_Protocol.ewma_filter(chunk, a)

        derivatives = IoTCupid_Protocol.compute_derivative(smoothed_data, window_size)
           
        received_events = IoTCupid_Protocol.detect_events(derivatives, bottom_th, top_th, agg_th)
     
        event_features = IoTCupid_Protocol.get_event_features(recieved_events, signal_data, feature_dim)

        for i in range(len(found_events)):
            recieved_events[i] = (
                recieved_events[i][0] + chunk_size * iteration,
                recieved_events[i][1] + chunk_size * iteration,
            )

        # Reconciling lumping adjacent events across windows
        if (
            len(received_events) != 0
            and len(events) != 0
            and received_events[0][0] - events[-1][1] <= lump_th
        ):
            events[-1] = (events[-1][0], received_events[0][1])
            length = events[-1][1] - events[-1][0] + 1
            max_amp = np.max([event_features[-1][1], received_event_features[0][1]])
            event_features[-1] = (length, max_amp)

            events.extend(received_events[1:])
            event_features.extend(received_event_features[1:])
        else:
            events.extend(received_events)
            event_features.extend(received_event_features)
        iteration += 1


def generate_bits(
    events,
    event_features,
    max_clusters,
    cluster_th,
    m_start,
    m_end,
    m_searches,
    quantization_factor,
    Fs,
    key_size_in_bytes,
):
    IoTCupid_Protocol.fuzzy_cmeans_w_elbow_method(
            event_features, max_clusters, cluster_th, m_start, m_end, m_searches
        )

    grouped_events = IoTCupid_Protocol.group_events(events, u)

    inter_event_timings = IoTCupid_Protocol.calculate_inter_event_timings(grouped_events)

    encoded_timings = IoTCupid_Protocol.encode_timings_to_bits(
        inter_event_timings, quantization_factor
    )

    return encoded_timings, grouped_events
