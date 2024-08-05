import os
import sys
import argparse

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src

from signal_processing.perceptio import PerceptioProcessing  # noqa: E402

goldsig_rng = np.random.default_rng(0)


def golden_signal(sample_num):
    return goldsig_rng.integers(0, 10, size=sample_num)


adv_rng = np.random.default_rng(12345)


def adversary_signal(sample_num):
    return adv_rng.integers(0, 10, size=sample_num)


def get_events(arr, top_th, bottom_th, lump_th, a):
    events = PerceptioProcessing.get_events(arr, a, bottom_th, top_th, lump_th)
    event_features = PerceptioProcessing.get_event_features(events, arr)

    return events, event_features


def gen_min_events(
    signal,
    chunk_size,
    min_events,
    top_th,
    bottom_th,
    lump_th,
    a,
):
    events = []
    event_features = []
    iteration = 0
    while len(events) < min_events:
        chunk = signal.read(chunk_size)

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
    labels, k = PerceptioProcessing.kmeans_w_elbow_method(
        event_features, cluster_sizes_to_check, cluster_th
    )

    grouped_events = PerceptioProcessing.group_events(events, labels, k)

    fps = PerceptioProcessing.gen_fingerprints(grouped_events, k, key_size_in_bytes, Fs)
    return fps, grouped_events



def get_command_line_args(
    top_threshold_default=6,
    bottom_threshold_default=4,
    lump_threshold_default=4,
    ewma_a_default=0.75,
    cluster_sizes_to_check_default=4,
    minimum_events_default=16,
    sampling_frequency_default=10000,
    chunk_size_default=10000,
    buffer_size_default=50000,
    key_length_default=128,
    snr_level_default=20,
    trials_default=100
):
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument("-tt", "--top_threshold", type=float, default=top_threshold_default)
    parser.add_argument("-bt", "--bottom_threshold", type=float, default=bottom_threshold_default)
    parser.add_argument("-lt", "--lump_threshold", type=int, default=lump_threshold_default)
    parser.add_argument("-a", "--ewma_a", type=float, default=ewma_a_default)
    parser.add_argument("-cl", "--cluster_sizes_to_check", type=int, default=cluster_sizes_to_check_default)
    parser.add_argument("-min", "--minimum_events", type=int, default=minimum_events_default)
    parser.add_argument("-fs", "--sampling_frequency", type=float, default=sampling_frequency_default)
    parser.add_argument("-ch", "--chunk_size", type=int, default=chunk_size_default)
    parser.add_argument("-bs", "--buffer_size", type=int, default=buffer_size_default)
    parser.add_argument("-kl", "--key_length", type=int, default=key_length_default)
    parser.add_argument("-snr", "--snr_level", type=float, default=snr_level_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    top_th = getattr(args, "top_threshold")
    bottom_th = getattr(args, "bottom_threshold")
    lump_th = getattr(args, "lump_threshold")
    a = getattr(args, "ewma_a")
    cluster_sizes_to_check = getattr(args, "cluster_sizes_to_check")
    cluster_th = 0.1  # Set a fixed cluster threshold
    min_events = getattr(args, "minimum_events")
    Fs = getattr(args, "sampling_frequency")
    chunk_size = getattr(args, "chunk_size")
    buffer_size = getattr(args, "buffer_size")
    key_size_in_bytes = getattr(args, "key_length") // 8
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return (top_th, bottom_th, lump_th, a, cluster_sizes_to_check, cluster_th,
            min_events, Fs, chunk_size, buffer_size, key_size_in_bytes, target_snr, trials)

