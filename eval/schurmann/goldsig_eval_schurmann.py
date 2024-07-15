import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    golden_signal,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)


def goldsig_eval(
    window_length,
    band_length,
    key_length,
    goldsig_sampling_freq,
    antialias_freq,
    trials,
):
    legit_bit_errs = []
    adv_bit_errs = []
    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, goldsig_sampling_freq, antialias_freq
    )
    signal = golden_signal(sample_num, goldsig_sampling_freq)
    adv_signal = adversary_signal(sample_num, goldsig_sampling_freq)
    for i in range(trials):
        bits1 = schurmann_wrapper_func(
            signal, window_length, band_length, goldsig_sampling_freq, antialias_freq
        )
        bits2 = schurmann_wrapper_func(
            signal, window_length, band_length, goldsig_sampling_freq, antialias_freq
        )
        adv_bits = schurmann_wrapper_func(
            adv_signal, window_length, band_length, goldsig_sampling_freq, antialias_freq
        )
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wl", "--window_length", type=int, default=16537)
    parser.add_argument("-bl", "--band_length", type=int, default=500)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    args = parser.parse_args()
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = goldsig_eval(
        window_length,
        band_length,
        key_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
        trials,
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")

