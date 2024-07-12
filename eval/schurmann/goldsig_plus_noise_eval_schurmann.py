import os
import sys
import argparse


sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    add_gauss_noise,
    golden_signal,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)


def goldsig_plus_noise_eval(
    window_length,
    band_length,
    key_length,
    goldsig_sampling_freq,
    antialias_freq,
    target_snr,
    trials,
):
    bit_errs = []
    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, goldsig_sampling_freq, antialias_freq
    )
    signal = golden_signal(sample_num, goldsig_sampling_freq)
    for i in range(trials):
        sig1 = add_gauss_noise(signal, target_snr)
        sig2 = add_gauss_noise(signal, target_snr)
        bits1 = schurmann_wrapper_func(
            sig1, window_length, band_length, goldsig_sampling_freq, antialias_freq
        )
        bits2 = schurmann_wrapper_func(
            sig2, window_length, band_length, goldsig_sampling_freq, antialias_freq
        )
        bit_err = cmp_bits(bits1, bits2, key_length)
        bit_errs.append(bit_err)
    return bit_errs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wl', '--window_length', type=int, default=16537)
    parser.add_argument('-bl', '--band_length', type=int, default=500)
    parser.add_argument('-kl', '--key_length', type=int, default=128)
    parser.add_argument('-snr', '--snr_level', type=float, default=20)
    parser.add_argument('-t', '--trials', type=int, default=1000)

    
    args = parser.parse_args()
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    bit_errs = goldsig_plus_noise_eval(
        window_length,
        band_length,
        key_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
        snr_level,
        trials,
    )
    print(f"Average Bit Error Rate: {np.mean(bit_errs)}")
