import argparse
import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src
from signal_processing.fastzip import FastZIPProcessing  # noqa: E402

SAMPLING_RATE = 50


def manage_overlapping_chunks(signal_buffer, chunk_size, overlap_size):
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
    sensor_arr,
    bits,
    power_thr,
    snr_thr,
    peak_thr,
    bias,
    sample_rate,
    eqd_delta,
    peak_status=None,
    ewma_filter=None,
    alpha=None,
    remove_noise=None,
    normalize=None,
):
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


def golden_signal(sample_num):
    np.random.seed(0)
    output = np.random.rand(sample_num)
    return output


def adversary_signal(sample_num):
    np.random.seed(12)
    output = np.random.rand(sample_num)
    return output



def parse_command_line_args(
    window_size_default=200,
    overlap_size_default=100,
    buffer_size_default=50000,
    n_bits_default=18,
    key_length_default=128,
    bias_default=0,
    eqd_delta_default=1,
    peak_status_default=None,
    ewma_filter_default=None,
    alpha_default=None,
    remove_noise_default=None,
    normalize_default=True,
    power_threshold_default=70,
    snr_threshold_default=1.2,
    number_peaks_default=0,
    snr_level_default=20,
    trials_default=1000
):
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
    parser.add_argument("-rn", "--remove-noise", type=bool, default=remove_noise_default)
    parser.add_argument("-n", "--normalize", type=bool, default=normalize_default)
    parser.add_argument("-pt", "--power-threshold", type=int, default=power_threshold_default)
    parser.add_argument("-st", "--snr-threshold", type=float, default=snr_threshold_default)
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

    return (window_size, overlap_size, buffer_size, n_bits, key_length, bias, eqd_delta, peak_status, ewma_filter,
            alpha, remove_noise, normalize, power_threshold, snr_threshold, number_peaks, snr_level, trials)