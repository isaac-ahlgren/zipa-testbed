# import math
import os
import sys

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Schurmann and Siggs algorithm in /src

import numpy as np

from protocols.miettinen import Miettinen_Protocol

MICROPHONE_SAMPLING_RATE = 48000


def miettinen_wrapper_func(arr, f, w, rel_thresh, abs_thresh):
    return Miettinen_Protocol.miettinen_algo(arr, f, w, rel_thresh, abs_thresh)


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


def add_gauss_noise(signal, target_snr):
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg), len(signal))
    return signal + noise
