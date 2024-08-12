import argparse
import os
import sys
from typing import Any, List, Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src

from signal_processing.iotcupid import IoTCupidProcessing  # noqa: E402

goldsig_rng = np.random.default_rng(0)


def golden_signal(sample_num):
    """
    Generate a golden signal with random integers between 0 and 10.

    :param sample_num: Number of samples to generate.
    :return: Array of random integers.
    """
    return goldsig_rng.integers(0, 10, size=sample_num)


adv_rng = np.random.default_rng(12345)


def adversary_signal(sample_num):
    """
    Generate an adversary signal with random integers between 0 and 10.

    :param sample_num: Number of samples to generate.
    :return: Array of random integers.
    """
    return adv_rng.integers(0, 10, size=sample_num)


def gen_min_events(
    signal: Any,
    chunk_size: int,
    min_events: int,
    top_th: float,
    bottom_th: float,
    agg_th: int,
    a: float,
    window_size: int,
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """
    Generate a minimum number of events from a signal based on specified thresholds.

    :param signal: Signal object from which data is read.
    :param chunk_size: Number of data points to read in each chunk.
    :param min_events: Minimum number of events required.
    :param top_th: Top threshold for event detection.
    :param bottom_th: Bottom threshold for event detection.
    :param agg_th: Aggregation threshold for lumping close events.
    :param a: EWMA parameter.
    :param window_size: Window size for computing derivatives.
    :return: A tuple of event indices and their corresponding signal data.
    """
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
    events: List[Tuple[int, int]],
    event_signals: List[np.ndarray],
    max_clusters: int,
    cluster_th: float,
    m_start: float,
    m_end: float,
    m_searches: int,
    quantization_factor: float,
    feature_dim: int,
    Fs: float,
    key_size_in_bytes: int,
    mem_th: float,
) -> Tuple[List[float], List[List[int]]]:
    """
    Generate bits from event data using fuzzy clustering and inter-event timing calculations.

    :param events: List of event start and end indices.
    :param event_signals: Corresponding signal data for each event.
    :param max_clusters: Maximum number of clusters to consider.
    :param cluster_th: Threshold for clustering.
    :param m_start: Start value for clustering searches.
    :param m_end: End value for clustering searches.
    :param m_searches: Number of searches for optimal clustering.
    :param quantization_factor: Factor for quantizing the inter-event timings.
    :param feature_dim: Dimensionality of the feature space for clustering.
    :param Fs: Sampling frequency.
    :param key_size_in_bytes: Key size in bytes.
    :param mem_th: Membership threshold for fuzzy clustering.
    :return: Tuple of inter-event timings and grouped events.
    """
    event_features = IoTCupidProcessing.get_event_features(event_signals, feature_dim)

    cntr, u, optimal_clusters, fpcs = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
        event_features.T, max_clusters, cluster_th, m_start, m_end, m_searches, mem_th
    )

    grouped_events = IoTCupidProcessing.group_events(events, u, mem_th)

    inter_event_timings = IoTCupidProcessing.calculate_inter_event_timings(
        grouped_events, Fs, quantization_factor, key_size_in_bytes
    )

    return inter_event_timings, grouped_events


def get_command_line_args(
    top_threshold_default: float = 0.07,
    bottom_threshold_default: float = 0.05,
    lump_threshold_default: int = 4,
    ewma_a_default: float = 0.75,
    cluster_sizes_to_check_default: int = 4,
    minimum_events_default: int = 16,
    sampling_frequency_default: int = 10000,
    chunk_size_default: int = 10000,
    buffer_size_default: int = 50000,
    window_size_default: int = 10,
    feature_dimensions_default: int = 3,
    quantization_factor_default: int = 1,
    mstart_default: int = 1.1,
    msteps_default: int = 10,
    mend_default: int = 2,
    key_length_default: int = 128,
    snr_level_default: int = 1,
    trials_default: int = 100,
) -> Tuple[
    float,
    float,
    int,
    float,
    int,
    int,
    float,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    int,
    float,
    int,
    float,
    int,
]:
    """
    Get and parse command-line arguments.

    :return: Tuple of parsed values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tt", "--top_threshold", type=float, default=top_threshold_default
    )
    parser.add_argument(
        "-bt", "--bottom_threshold", type=float, default=bottom_threshold_default
    )
    parser.add_argument(
        "-lt", "--lump_threshold", type=int, default=lump_threshold_default
    )
    parser.add_argument("-a", "--ewma_a", type=float, default=ewma_a_default)
    parser.add_argument(
        "-cl",
        "--cluster_sizes_to_check",
        type=int,
        default=cluster_sizes_to_check_default,
    )
    parser.add_argument(
        "-min", "--minimum_events", type=int, default=minimum_events_default
    )
    parser.add_argument(
        "-fs", "--sampling_frequency", type=float, default=sampling_frequency_default
    )
    parser.add_argument("-ch", "--chunk_size", type=int, default=chunk_size_default)
    parser.add_argument("-bs", "--buffer_size", type=int, default=buffer_size_default)
    parser.add_argument("-ws", "--window_size", type=int, default=window_size_default)
    parser.add_argument(
        "-fd", "--feature_dimensions", type=int, default=feature_dimensions_default
    )
    parser.add_argument(
        "-w", "--quantization_factor", type=float, default=quantization_factor_default
    )
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

    return (
        top_th,
        bottom_th,
        lump_th,
        a,
        cluster_sizes_to_check,
        cluster_th,
        min_events,
        Fs,
        chunk_size,
        buffer_size,
        window_size,
        feature_dimensions,
        w,
        m_start,
        m_steps,
        m_end,
        key_size_in_bytes,
        target_snr,
        trials,
    )
