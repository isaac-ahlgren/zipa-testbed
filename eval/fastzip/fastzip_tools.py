import argparse
import os
import sys
from typing import Any, Generator, List, Optional, Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src
from signal_processing.fastzip import FastZIPProcessing  # noqa: E402

SAMPLING_RATE = 50


def manage_overlapping_chunks(
    signal_buffer: np.ndarray, chunk_size: int, overlap_size: int
) -> Generator[np.ndarray, None, None]:
    """
    Generate overlapping chunks from a signal buffer.

    :param signal_buffer: Input signal data as a numpy array.
    :param chunk_size: Size of each chunk to be processed.
    :param overlap_size: Number of overlapping samples between consecutive chunks.
    :return: Yields successive overlapping chunks from the signal buffer.
    """
    previous_chunk = np.array([])

    while True:
        if len(previous_chunk) < overlap_size:
            new_data = signal_buffer.read(chunk_size)
            if new_data is None:
                break
        else:
            new_data = signal_buffer.read(chunk_size - overlap_size)

        if new_data is None:
            break

        if len(previous_chunk) >= overlap_size:
            current_chunk = np.concatenate((previous_chunk[-overlap_size:], new_data))
        else:
            current_chunk = new_data

        yield current_chunk
        previous_chunk = current_chunk


def fastzip_wrapper_function(
    sensor_arr: np.ndarray,
    bits: int,
    power_thr: float,
    snr_thr: float,
    peak_thr: int,
    bias: int,
    sample_rate: int,
    eqd_delta: int,
    peak_status: Optional[bool] = None,
    ewma_filter: Optional[float] = None,
    alpha: Optional[float] = None,
    remove_noise: Optional[bool] = None,
    normalize: Optional[bool] = None,
) -> List[int]:
    """
    Wrapper function to call the FastZIP processing algorithm.

    :param sensor_arr: Array of sensor data.
    :param bits: Number of bits to process.
    :param power_thr: Power threshold.
    :param snr_thr: Signal-to-noise ratio threshold.
    :param peak_thr: Peak threshold.
    :param bias: Bias adjustment for processing.
    :param sample_rate: Sampling rate of the signal.
    :param eqd_delta: Equalization delta for signal processing.
    :param peak_status: Peak status for processing (optional).
    :param ewma_filter: EWMA filter value (optional).
    :param alpha: Alpha value for processing (optional).
    :param remove_noise: Flag to indicate noise removal (optional).
    :param normalize: Flag to indicate whether to normalize the signal (optional).
    :return: Processed result as a list of integers.
    """
    return FastZIPProcessing.fastzip_algo(
        [sensor_arr],
        [bits],
        [power_thr],
        [snr_thr],
        [peak_thr],
        [bias],
        [sample_rate],
        [eqd_delta],
        [peak_status],
        [ewma_filter],
        [alpha],
        [remove_noise],
        [normalize],
    )


def golden_signal(sample_num: int) -> np.ndarray:
    """
    Generate a golden signal using a predefined seed.

    :param sample_num: Number of samples in the generated signal.
    :return: Golden signal as a numpy array.
    """
    np.random.seed(0)
    output = np.random.rand(sample_num)
    return output


def adversary_signal(sample_num: int) -> np.ndarray:
    """
    Generate an adversary signal using a predefined seed.

    :param sample_num: Number of samples in the generated signal.
    :return: Adversary signal as a numpy array.
    """
    np.random.seed(12)
    output = np.random.rand(sample_num)
    return output


def parse_command_line_args(
    window_size_default: int = 200,
    overlap_size_default: int = 100,
    buffer_size_default: int = 50000,
    n_bits_default: int = 18,
    key_length_default: int = 128,
    bias_default: int = 0,
    eqd_delta_default: int = 1,
    peak_status_default: Any | None = None,
    ewma_filter_default: Any | None = None,
    alpha_default: Any | None = None,
    remove_noise_default: Any | None = None,
    normalize_default: bool = True,
    power_threshold_default: int = 70,
    snr_threshold_default: float = 1.2,
    number_peaks_default: int = 0,
    snr_level_default: int = 20,
    trials_default: int = 1000,
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    Optional[bool],
    Optional[float],
    Optional[float],
    Optional[bool],
    Optional[bool],
    float,
    float,
    int,
    int,
    int,
]:
    """
    Parse command line arguments for the application using argparse.

    :return: A tuple containing all parsed values.
    """
    parser = argparse.ArgumentParser()

    # Add arguments with descriptions
    parser.add_argument("-ws", "--window-size", type=int, default=window_size_default)
    parser.add_argument("-os", "--overlap-size", type=int, default=overlap_size_default)
    parser.add_argument("-bs", "--buffer-size", type=int, default=buffer_size_default)
    parser.add_argument("-nb", "--n-bits", type=int, default=n_bits_default)
    parser.add_argument("-kl", "--key-length", type=int, default=key_length_default)
    parser.add_argument("-b", "--bias", type=int, default=bias_default)
    parser.add_argument("-ed", "--eqd-delta", type=int, default=eqd_delta_default)
    parser.add_argument("-ps", "--peak-status", type=bool, default=peak_status_default)
    parser.add_argument("-ef", "--ewma-filter", type=bool, default=ewma_filter_default)
    parser.add_argument("-a", "--alpha", type=float, default=alpha_default)
    parser.add_argument(
        "-rn", "--remove-noise", type=bool, default=remove_noise_default
    )
    parser.add_argument("-n", "--normalize", type=bool, default=normalize_default)
    parser.add_argument(
        "-pt", "--power-threshold", type=int, default=power_threshold_default
    )
    parser.add_argument(
        "-st", "--snr-threshold", type=float, default=snr_threshold_default
    )
    parser.add_argument("-np", "--number-peaks", type=int, default=number_peaks_default)
    parser.add_argument("-snr", "--snr-level", type=int, default=snr_level_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    window_size = getattr(args, "window_size")
    overlap_size = getattr(args, "overlap_size")
    buffer_size = getattr(args, "buffer_size")
    n_bits = getattr(args, "n_bits")
    key_length = getattr(args, "key_length")
    bias = getattr(args, "bias")
    eqd_delta = getattr(args, "eqd_delta")
    peak_status = getattr(args, "peak_status")
    ewma_filter = getattr(args, "ewma_filter")
    alpha = getattr(args, "alpha")
    remove_noise = getattr(args, "remove_noise")
    normalize = getattr(args, "normalize")
    power_threshold = getattr(args, "power_threshold")
    snr_threshold = getattr(args, "snr_threshold")
    number_peaks = getattr(args, "number_peaks")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return (
        window_size,
        overlap_size,
        buffer_size,
        n_bits,
        key_length,
        bias,
        eqd_delta,
        peak_status,
        ewma_filter,
        alpha,
        remove_noise,
        normalize,
        power_threshold,
        snr_threshold,
        number_peaks,
        snr_level,
        trials,
    )
