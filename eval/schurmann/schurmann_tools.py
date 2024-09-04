import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Schurmann and Siggs algorithm in /src

from signal_processing.shurmann import SchurmannProcessing  # noqa: E402

MICROPHONE_SAMPLING_RATE = 48000
ANTIALIASING_FILTER = 18000
DATA_DIRECTORY = "./schurmann_data"

def schurmann_wrapper_func(
    arr: np.ndarray,
    window_length: int,
    band_len: int,
    sampling_freq: int,
    antialias_freq: int,
) -> np.ndarray:
    """
    Wrapper function to process an array using Schurmann and Siggs algorithm to reduce aliasing.

    :param arr: Input signal array.
    :param window_length: Length of the analysis window.
    :param band_len: Number of frequency bands to consider.
    :param sampling_freq: Sampling frequency of the input signal.
    :param antialias_freq: Frequency threshold for anti-aliasing.
    :return: Processed signal array with reduced aliasing.
    """
    return SchurmannProcessing.zero_out_antialias_sigs_algo(
        arr, antialias_freq, sampling_freq, window_len=window_length, bands=band_len
    )


def golden_signal(sample_num: int, fs: int) -> np.ndarray:
    """
    Generate a golden signal as a simple sinusoidal wave.

    :param sample_num: Number of samples to generate.
    :param fs: Sampling frequency of the signal.
    :return: Sinusoidal signal array.
    """
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * np.pi / fs * i)
    return output


def adversary_signal(sample_num: int, fs: int) -> np.ndarray:
    """
    Generate an adversary sinusoidal signal with a higher base frequency.

    :param sample_num: Number of samples in the signal.
    :param fs: Sampling frequency.
    :return: Sinusoidal signal array at higher frequency.
    """
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * (8000) * np.pi / fs * i)
    return output


def schurmann_calc_sample_num(
    key_length: int,
    window_length: int,
    band_length: int,
    sampling_freq: int,
    antialias_freq: int,
) -> int:
    """
    Calculate the number of samples required based on the key length and other parameters.

    :param key_length: Desired key length.
    :param window_length: Length of the window used in the process.
    :param band_length: Number of frequency bands.
    :param sampling_freq: Sampling frequency.
    :param antialias_freq: Anti-aliasing filter frequency.
    :return: Calculated number of samples.
    """
    freq_bin_len = (sampling_freq / 2) / (int(window_length / 2) + 1)
    antialias_bin = int(np.floor(antialias_freq / freq_bin_len))
    return (
        math.ceil(((key_length) / int(antialias_bin / band_length)) + 1) * window_length
    )


def get_command_line_args(
    window_length_default: int = 16537,
    band_length_default: int = 500,
    key_length_default: int = 128,
    snr_level_default: float = 20,
    trials_default: int = 1000,
) -> Tuple[int, int, int, float, int]:
    """
    Parse command-line arguments for the script.

    :return: Tuple containing window length, band length, key length, SNR level, and number of trials.
    """
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument(
        "-wl", "--window_length", type=int, default=window_length_default
    )
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
