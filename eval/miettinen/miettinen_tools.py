# import math
import argparse
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Miettinen algorithm in /src

from signal_processing.miettinen import MiettinenProcessing  # noqa: E402

MICROPHONE_SAMPLING_RATE = 48000


def miettinen_wrapper_func(
    arr: np.ndarray, f: float, w: float, rel_thresh: float, abs_thresh: float
) -> np.ndarray:
    """
    Wrapper function to process an array with the Miettinen algorithm.

    :param arr: Input signal array.
    :param f: Frequency bandwidth to analyze.
    :param w: Snapshot width of the signal.
    :param rel_thresh: Relative threshold for the processing.
    :param abs_thresh: Absolute threshold for the processing.
    :return: Processed signal array.
    """
    return MiettinenProcessing.miettinen_algo(arr, f, w, rel_thresh, abs_thresh)


def golden_signal(sample_num: int, fs: int) -> np.ndarray:
    """
    Generate a golden signal based on sinusoidal function.

    :param sample_num: Number of samples to generate.
    :param fs: Sampling frequency.
    :return: Sinusoidal signal array.
    """
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * np.pi / fs * i) + 1
    return output


def adversary_signal(sample_num: int, fs: int) -> np.ndarray:
    """
    Generate an adversary signal based on sinusoidal function at a higher frequency.

    :param sample_num: Number of samples to generate.
    :param fs: Sampling frequency.
    :return: Sinusoidal signal array at higher frequency.
    """
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * (8000) * np.pi / fs * i)
    return output


def miettinen_calc_sample_num(key_length: int, w: float, f: float) -> int:
    """
    Calculate the number of samples needed for the Miettinen algorithm based on key length.

    :param key_length: Desired key length.
    :param w: Snapshot width of the signal.
    :param f: Frequency bandwidth.
    :return: Number of samples needed.
    """
    return (w + f) * (key_length + 1)


def get_command_line_args(
    snap_shot_width_default: int = 5,
    no_snap_shot_width_default: int = 5,
    absolute_threshold_default: float = 5e-15,
    relative_threshold_default: float = 0.1,
    key_length_default: int = 128,
    snr_level_default: int = 20,
    trials_default: int = 100,
) -> Tuple[int, int, float, float, int, int, int]:
    """
    Parse command-line arguments for the Miettinen processing algorithm.

    :return: Tuple containing parsed command-line argument values.
    """
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument(
        "-w", "--snap_shot_width", type=float, default=snap_shot_width_default
    )
    parser.add_argument(
        "-f", "--no_snap_shot_width", type=float, default=no_snap_shot_width_default
    )
    parser.add_argument(
        "-at", "--absolute_threshold", type=float, default=absolute_threshold_default
    )
    parser.add_argument(
        "-rt", "--relative_threshold", type=float, default=relative_threshold_default
    )
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

    return (
        snap_shot_width,
        no_snap_shot_width,
        absolute_threshold,
        relative_threshold,
        key_length,
        snr_level,
        trials,
    )
