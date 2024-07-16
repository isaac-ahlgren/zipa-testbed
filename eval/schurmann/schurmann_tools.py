import math
import os
import sys

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Schurmann and Siggs algorithm in /src

import numpy as np

from protocols.shurmann import Shurmann_Siggs_Protocol

MICROPHONE_SAMPLING_RATE = 48000
ANTIALIASING_FILTER = 18000


def schurmann_wrapper_func(arr, window_length, band_len, sampling_freq, antialias_freq):
    return Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(
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


def add_gauss_noise(signal, target_snr):
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg), len(signal))
    return signal + noise
