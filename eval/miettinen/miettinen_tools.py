# import math
import os
import sys
import argparse

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Miettinen algorithm in /src

from signal_processing.miettinen import MiettinenProcessing  # noqa: E402

MICROPHONE_SAMPLING_RATE = 48000


def miettinen_wrapper_func(arr, f, w, rel_thresh, abs_thresh):
    return MiettinenProcessing.miettinen_algo(arr, f, w, rel_thresh, abs_thresh)


def golden_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * np.pi / fs * i) + 1
    return output


def adversary_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * (8000) * np.pi / fs * i)
    return output


def miettinen_calc_sample_num(key_length, w, f):
    return (w + f) * (key_length + 1)


import argparse

def get_command_line_args(
    snap_shot_width_default,
    no_snap_shot_width_default,
    absolute_threshold_default,
    relative_threshold_default,
    key_length_default,
    snr_level_default,
    trials_default
):
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument("-w", "--snap_shot_width", type=float, default=snap_shot_width_default)
    parser.add_argument("-f", "--no_snap_shot_width", type=float, default=no_snap_shot_width_default)
    parser.add_argument("-at", "--absolute_threshold", type=float, default=absolute_threshold_default)
    parser.add_argument("-rt", "--relative_threshold", type=float, default=relative_threshold_default)
    parser.add_argument("-kl", "--key_length", type=int, default=key_length_default)
    parser.add_argument("-snr", "--snr_level", type=float, default=snr_level_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    snap_shot_width = getattr(args, "snap_shot_width")
    no_snap_shot_width = getattr(args, "no_snap_shot_width")
    absolute_threshold = getattr(args, "absolute_threshold")
    relative_threshold = getattr(args, "relative_threshold")
    key_length = getattr(args, "key_length")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return (snap_shot_width, no_snap_shot_width, absolute_threshold,
            relative_threshold, key_length, snr_level, trials)
