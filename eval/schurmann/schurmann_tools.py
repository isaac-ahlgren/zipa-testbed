import math
import os
import sys
import argparse

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Schurmann and Siggs algorithm in /src

from signal_processing.shurmann import SchurmannProcessing  # noqa: E402

MICROPHONE_SAMPLING_RATE = 48000
ANTIALIASING_FILTER = 18000


def schurmann_wrapper_func(arr, window_length, band_len, sampling_freq, antialias_freq):
    return SchurmannProcessing.zero_out_antialias_sigs_algo(
        arr, antialias_freq, sampling_freq, window_len=window_length, bands=band_len
    )


def golden_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * np.pi / fs * i)
    return output


def adversary_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * (8000) * np.pi / fs * i)
    return output


def schurmann_calc_sample_num(
    key_length, window_length, band_length, sampling_freq, antialias_freq
):
    freq_bin_len = (sampling_freq / 2) / (int(window_length / 2) + 1)
    antialias_bin = int(np.floor(antialias_freq / freq_bin_len))
    return (
        math.ceil(((key_length) / int(antialias_bin / band_length)) + 1) * window_length
    )


def get_command_line_args(
    window_length_default=16537,
    band_length_default=500,
    key_length_default=128,
    snr_level_default=20,
    trials_default=1000
):
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument("-wl", "--window_length", type=int, default=window_length_default)
    parser.add_argument("-bl", "--band_length", type=int, default=band_length_default)
    parser.add_argument("-kl", "--key_length", type=int, default=key_length_default)
    parser.add_argument("-snr", "--snr_level", type=float, default=snr_level_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return window_length, band_length, key_length, target_snr, trials