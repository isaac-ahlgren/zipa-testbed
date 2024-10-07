import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Perceptio algorithm in /src

from signal_processing.perceptio import PerceptioProcessing  # noqa: E402

MICROPHONE_SAMPLING_RATE = 44100
DATA_DIRECTORY = "./perceptio_data"

goldsig_rng = np.random.default_rng(0)


def golden_signal(sample_num: int) -> np.ndarray:
    """
    Generate a golden signal with random integers between 0 and 10.

    :param sample_num: Number of samples to generate.
    :return: Array of random integers.
    """
    return goldsig_rng.integers(0, 10, size=sample_num)


adv_rng = np.random.default_rng(12345)


def adversary_signal(sample_num: int) -> np.ndarray:
    """
    Generate an adversary signal with random integers between 0 and 10.

    :param sample_num: Number of samples to generate.
    :return: Array of random integers.
    """
    return adv_rng.integers(0, 10, size=sample_num)


def get_events(arr, top_th, bottom_th, lump_th, a):
    return PerceptioProcessing.get_events(arr, a, bottom_th, top_th, lump_th)

def merge_events(first_event_list, second_event_list, lump_th, chunk_size, iteration):
    for i in range(len(second_event_list)):
        second_event_list[i] = (
                second_event_list[i][0] + iteration*chunk_size,
                second_event_list[i][1] + iteration*chunk_size,
            )

    event_list = []
    if len(first_event_list) != 0 and len(second_event_list) != 0:
        end_event = first_event_list[-1]
        beg_event = second_event_list[0]

        if beg_event[0] - end_event[1] < lump_th:
            new_event = (end_event[0], beg_event[1])
            event_list.extend(first_event_list[:-1])
            event_list.append(new_event)
            event_list.extend(second_event_list[1:])
        else:
            event_list.extend(first_event_list)
            event_list.extend(second_event_list)
    else:
        event_list.extend(first_event_list)
        event_list.extend(second_event_list)
    
    return event_list

def process_events(events, event_signals, cluster_sizes_to_check, cluster_th, key_size, Fs):

    event_features = [PerceptioProcessing.generate_features(x) for x in event_signals]

    labels, k, _ = PerceptioProcessing.kmeans_w_elbow_method(
        event_features, cluster_sizes_to_check, cluster_th
    )

    grouped_events = PerceptioProcessing.group_events(events, labels, k)

    fps = PerceptioProcessing.gen_fingerprints(grouped_events, k, key_size, Fs)

    return fps

def extract_all_events(signal, top_th, bottom_th, lump_th, a, chunk_size=10000):
    events = None
    iteration = 0
    while not signal.get_finished_reading():
        chunk = signal.read(chunk_size)
        new_events = get_events(chunk, top_th, bottom_th, lump_th, a)
        if events is not None:
            events = merge_events(events, new_events, lump_th, chunk_size, iteration)
        else:
            events = new_events
        iteration += 1
    return events

def extract_events(
    arr: np.ndarray, top_th: float, bottom_th: float, lump_th: int, a: float
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """
    Retrieve events and their features from a signal array based on the Perceptio algorithm.

    :param arr: Input signal array.
    :param top_th: Top threshold for event detection.
    :param bottom_th: Bottom threshold for event detection.
    :param lump_th: Lump threshold for grouping events.
    :param a: Parameter for the EWMA calculation.
    :return: A tuple of events and their corresponding features.
    """
    events = PerceptioProcessing.get_events(arr, a, bottom_th, top_th, lump_th)
    event_features = PerceptioProcessing.get_event_features(events, arr)

    return events, event_features

def gen_min_events(
    signal: np.ndarray,
    chunk_size: int,
    min_events: int,
    top_th: float,
    bottom_th: float,
    lump_th: int,
    a: float,
) -> Tuple[List[tuple], List[np.ndarray]]:
    """
    Generate a minimum number of events from a continuous signal by processing it in chunks.

    :param signal: Signal data to process.
    :param chunk_size: Size of each chunk to process.
    :param min_events: Minimum number of events to detect.
    :param top_th: Top threshold for event detection.
    :param bottom_th: Bottom threshold for event detection.
    :param lump_th: Lump threshold for grouping events.
    :param a: Parameter for the EWMA calculation.
    :return: A tuple containing lists of events and their features.
    """
    events = []
    event_features = []
    iteration = 0
    while len(events) < min_events:
        chunk = signal.read(chunk_size)

        found_events, found_event_features = extract_events(
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
    events: List[tuple],
    event_features: List[np.ndarray],
    cluster_sizes_to_check: int,
    cluster_th: float,
    Fs: float,
    key_size_in_bytes: int,
) -> Tuple[List[np.ndarray], List[list]]:
    """
    Generate bits (fingerprints) from detected events using clustering.

    :param events: Detected events.
    :param event_features: Features of the detected events.
    :param cluster_sizes_to_check: Number of cluster sizes to consider.
    :param cluster_th: Threshold for the clustering algorithm.
    :param Fs: Sampling frequency of the signal.
    :param key_size_in_bytes: Size of the generated key in bytes.
    :return: A tuple containing fingerprints and grouped events.
    """
    labels, k, inertias = PerceptioProcessing.kmeans_w_elbow_method(
        event_features, cluster_sizes_to_check, cluster_th
    )

    grouped_events = PerceptioProcessing.group_events(events, labels, k)

    fps = PerceptioProcessing.gen_fingerprints(grouped_events, k, key_size_in_bytes, Fs)
    return fps, grouped_events


def get_command_line_args(
    top_threshold_default: int = 6,
    bottom_threshold_default: int = 4,
    lump_threshold_default: int = 4,
    ewma_a_default: float = 0.75,
    cluster_sizes_to_check_default: int = 4,
    minimum_events_default: int = 16,
    sampling_frequency_default: int = 10000,
    chunk_size_default: int = 10000,
    buffer_size_default: int = 50000,
    key_length_default: int = 128,
    snr_level_default: int = 20,
    trials_default: int = 100,
) -> Tuple[int, int, int, float, int, int, int, int, int, int, int, int, int]:
    """
    Parse command line arguments for the script.

    :return: Tuple of parsed values as specified in the script.
    """
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
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
        key_size_in_bytes,
        target_snr,
        trials,
    )
