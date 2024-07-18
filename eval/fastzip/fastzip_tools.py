import math
import os
import sys

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src

import numpy as np

from protocols.fastzip import FastZIP_Protocol

SAMPLING_RATE = 100

def fastzip_wrapper_function(sensor_arr, bits, power_thr, snr_thr, peak_thr, bias):
    return FastZIP_Protocol.fastzip_algo(sensor_arr, bits, power_thr, snr_thr, peak_thr, bias)


def grab_parameters(sig, sample_rate):
    sig_copy = np.copy(sig)
    power_thr = FastZIP_Protocol.compute_sig_power(sig_copy)
    snr_thr = FastZIP_Protocol.compute_snr(sig_copy)
    peaks = FastZIP_Protocol.get_peaks(sig_copy, sample_rate)
    return power_thr, snr_thr, peaks

    
#Change the golden signal to output random numbers given a seed
#seed golden_signal with 0, seed adversarial with 12

def golden_signal(sample_num, seed):
    np.random.seed(seed)
    output = np.random.rand(sample_num)
    return output

def adversary_signal(sample_num, seed):
    np.random.seed(seed)
    output = np.random.rand(sample_num)
    return output


def fastzip_calc_sample_num(sample_rate, duration, w, f, n_bits):
    total_samples = sample_rate * duration
    effective_window = w + f - 1  # Adjust for overlap
    num_windows = math.ceil(total_samples / effective_window)  # Total number of windows that can fit in the sample space
    return num_windows * n_bits  # Total number of bits that can be processed/extracted

def add_gauss_noise(signal, target_snr):
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg), len(signal))
    return signal + noise
