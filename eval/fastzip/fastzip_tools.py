import math
import os
import sys

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src

import numpy as np

from protocols.fastzip import FastZIP_Protocol

SAMPLING_RATE = 100

def fastzip_wrapper_function(sensor_arr, bits, power_thr, snr_thr, peak_thr, bias, sample_rate, eqd_delta):
    return FastZIP_Protocol.fastzip_algo(
        [sensor_arr], [bits], [power_thr], [snr_thr], [peak_thr], [bias], [sample_rate], [eqd_delta]
    )

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


def fastzip_calc_sample_num(key_length, window_length):
    return (key_length + 1) * window_length

def add_gauss_noise(signal, target_snr):
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg), len(signal))
    return signal + noise
