import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py

import numpy as np
from eval_tools import cmp_bits

from fastzip_tools import (
    SAMPLING_RATE,
    add_gauss_noise,
    adversary_signal,
    golden_signal,
    fastzip_calc_sample_num,
    fastzip_wrapper_function,
)
from scipy.io import wavfile

#rewrite to work with fastzip
def controlled_sig_plus_noise_eval(
    window_length,
    band_length,
    key_length,
    sampling_freq,
    bias,
    eqd_delta,
    ewma_filter,
    alpha,
    remove_noise,
    normalize,
    power_thr,
    snr_thr,
    peaks,
    trials
):
    legit_bit_errs = []
    adv_bit_errs = []
    
    signal, sr = load_controlled_signal("../controlled_signal.wav")
    adv_signal, sr = load_controlled_signal("../adversary_controlled_signal.wav")
    #Need to improve upon how sample number is calculated
    sample_num = fastzip_calc_sample_num(key_length, window_length)
    index = 0
    adv_index = 0
    for i in range(trials):
        signal_part, index = wrap_around_read(signal, index, sample_num)
        adv_part, adv_index = wrap_around_read(adv_signal, adv_index, sample_num)
        sig1 = add_gauss_noise(signal_part, snr_thr)
        sig2 = add_gauss_noise(signal_part, snr_thr)
        adv_sig = add_gauss_noise(adv_part, snr_thr)

        bits1 = fastzip_wrapper_function(sig1, key_length, power_thr, snr_thr, peaks, bias, sampling_freq, eqd_delta, ewma_filter, alpha, remove_noise, normalize)
        bits2 = fastzip_wrapper_function(sig2, key_length, power_thr, snr_thr, peaks, bias, sampling_freq, eqd_delta, ewma_filter, alpha, remove_noise, normalize)
        adv_bits = fastzip_wrapper_function(adv_sig, key_length, power_thr, snr_thr, peaks, bias, sampling_freq, eqd_delta, ewma_filter, alpha, remove_noise, normalize)
        
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs


def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64) + 2**16, sr


def wrap_around_read(buffer, index, samples_to_read):
    output = np.array([])
    while samples_to_read != 0:
        samples_can_read = len(buffer) - index
        if samples_can_read <= samples_to_read:
            buf = buffer[index : index + samples_can_read]
            output = np.append(output, buf)
            samples_to_read = samples_to_read - samples_can_read
            index = 0
        else:
            buf = buffer[index : index + samples_to_read]
            output = np.append(output, buf)
            index = index + samples_to_read
            samples_to_read = 0
    return output, index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wl", "--window_length", type=int, default=50)
    parser.add_argument("-bl", "--band_length", type=int, default=500)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-b", "--bias", type=int, default=0)
    parser.add_argument("-ed", "--eqd_delta", type=int, default=1)
    parser.add_argument("-ef", "--ewma_filter", type=bool, default=None)
    parser.add_argument("-a", "--alpha", type=float, default=None)
    parser.add_argument("-rn", "--remove_noise", type=bool, default=None)
    parser.add_argument("-n", "--normalize", type=bool, default=None)
    parser.add_argument("-pt", "--power_threshold", type=int, default=-12)
    parser.add_argument("-st", "--snr_threshold", type=int, default=1.2)
    parser.add_argument("-np", "--number_peaks", type=int, default=14)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    args = parser.parse_args()
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    bias = getattr(args, "bias")
    eqd_delta = getattr(args, "eqd_delta")
    ewma_filter = getattr(args, "ewma_filter")
    alpha = getattr(args, "alpha")
    remove_noise = getattr(args, "remove_noise")
    normalize = getattr(args, "normalize")
    power_threshold = getattr(args, "power_threshold")
    snr_threshold = getattr(args, "snr_threshold")
    number_peaks = getattr(args, "number_peaks")
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = controlled_sig_plus_noise_eval(
        window_length,
        band_length,
        key_length,
        SAMPLING_RATE,
        bias,
        eqd_delta,
        ewma_filter,
        alpha,
        remove_noise,
        normalize,
        power_threshold,
        snr_threshold,
        number_peaks,
        trials
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")

