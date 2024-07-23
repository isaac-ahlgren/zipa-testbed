import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src
from protocols.perceptio import Perceptio_Protocol  # noqa: E402


def get_events(arr, top_th, bottom_th, lump_th, a):
    events = Perceptio_Protocol.get_events(arr, a, bottom_th, top_th, lump_th)
    event_features = Perceptio_Protocol.get_event_features(events, arr)

    return events, event_features


def gen_min_events(
    signal,
    chunk_size,
    min_events,
    top_th,
    bottom_th,
    lump_th,
    a,
    add_noise=False,
    snr=20,
):
    events = []
    event_features = []
    iteration = 0
    while len(events) < min_events:
        chunk = signal.read(chunk_size)

        if add_noise is True:
            chunk = add_gauss_noise(chunk, snr)
        found_events, found_event_features = get_events(
            chunk, top_th, bottom_th, lump_th, a
        )

        for i in range(len(found_events)):
            found_events[i] = (
                found_events[i][0] + chunk_size * iteration,
                found_events[i][1] + chunk_size * iteration,
            )

        events.extend(found_events)
        event_features.extend(found_event_features)
        iteration += 1
    return events, event_features


def generate_bits(
    events,
    event_features,
    cluster_sizes_to_check,
    cluster_th,
    Fs,
    key_size_in_bytes,
):
    labels, k = Perceptio_Protocol.kmeans_w_elbow_method(
        event_features, cluster_sizes_to_check, cluster_th
    )

    grouped_events = Perceptio_Protocol.group_events(events, labels, k)

    fps = Perceptio_Protocol.gen_fingerprints(grouped_events, k, key_size_in_bytes, Fs)
    return fps, grouped_events


def add_gauss_noise(signal, target_snr):
    sig_avg_sqr = np.mean(signal) ** 2
    sig_avg_db = 10 * np.log10(sig_avg_sqr)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg_sqr = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg_sqr), len(signal))
    return signal + noise
