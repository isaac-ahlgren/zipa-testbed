import os
import argparse
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
):
    event_features = IoTCupidProcessing.get_event_features(event_signals, feature_dim)

    cntr, u, optimal_clusters, fpcs = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        event_features.T, max_clusters, cluster_th, m_start, m_end, m_searches
    )

    grouped_events = IoTCupidProcessing.group_events(events, u)

    inter_event_timings = IoTCupidProcessing.calculate_inter_event_timings(
        grouped_events, Fs, quantization_factor, key_size_in_bytes
    )

    return inter_event_timings, grouped_events

def get_command_line_args(
    top_threshold_default,
    bottom_threshold_default,
    lump_threshold_default,
    ewma_a_default,
    cluster_sizes_to_check_default,
    minimum_events_default,
    sampling_frequency_default,
    chunk_size_default,
    buffer_size_default,
    window_size_default,
    feature_dimensions_default,
    quantization_factor_default,
    mstart_default,
    msteps_default,
    mend_default,
    key_length_default,
    snr_level_default,
    trials_default
):
    parser = argparse.ArgumentParser()
    parser.add_argument("-tt", "--top_threshold", type=float, default=top_threshold_default)
    parser.add_argument("-bt", "--bottom_threshold", type=float, default=bottom_threshold_default)
    parser.add_argument("-lt", "--lump_threshold", type=int, default=lump_threshold_default)
    parser.add_argument("-a", "--ewma_a", type=float, default=ewma_a_default)
    parser.add_argument("-cl", "--cluster_sizes_to_check", type=int, default=cluster_sizes_to_check_default)
    parser.add_argument("-min", "--minimum_events", type=int, default=minimum_events_default)
    parser.add_argument("-fs", "--sampling_frequency", type=float, default=sampling_frequency_default)
    parser.add_argument("-ch", "--chunk_size", type=int, default=chunk_size_default)
    parser.add_argument("-bs", "--buffer_size", type=int, default=buffer_size_default)
    parser.add_argument("-ws", "--window_size", type=int, default=window_size_default)
    parser.add_argument("-fd", "--feature_dimensions", type=int, default=feature_dimensions_default)
    parser.add_argument("-w", "--quantization_factor", type=float, default=quantization_factor_default)
    parser.add_argument("-mstart", "--mstart", type=float, default=mstart_default)
    parser.add_argument("-msteps", "--msteps", type=int, default=msteps_default)
    parser.add_argument("-mend", "--mend", type=float, default=mend_default)
    parser.add_argument("-kl", "--key_length", type=int, default=key_length_default)
    parser.add_argument("-snr", "--snr_level", type=float, default=snr_level_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()
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
    window_size = getattr(args, "window_size")
    feature_dimensions = getattr(args, "feature_dimensions")
    w = getattr(args, "quantization_factor")
    m_start = getattr(args, "mstart")
    m_steps = getattr(args, "msteps")
    m_end = getattr(args, "mend")
    key_size_in_bytes = getattr(args, "key_length") // 8
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return top_th, bottom_th, lump_th, a, cluster_sizes_to_check, cluster_th, min_events, Fs, chunk_size, buffer_size, window_size, feature_dimensions, w, m_start, m_steps, m_end, key_size_in_bytes, target_snr, trials
