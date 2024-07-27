import argparse
import os
import sys

import numpy as np
from fastzip_tools import (
    SAMPLING_RATE,
    adversary_signal,
    fastzip_calc_sample_num,
    fastzip_wrapper_function,
    golden_signal,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

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

    sample_num = fastzip_calc_sample_num(key_length, window_length)
    signal1 = golden_signal(sample_num, seed=0)
    signal2 = golden_signal(sample_num, seed=0)
    adv_signal = adversary_signal(sample_num, seed=12)
    signals = (signal1, signal2, adv_signal)

    def bit_gen_algo(signal):
        return fastzip_wrapper_function(
            signal,
            key_length,
            power_threshold,
            snr_threshold,
            number_peaks,
            bias,
            SAMPLING_RATE,
            eqd_delta,
            ewma_filter,
            alpha,
            remove_noise,
            normalize,
        )

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)

    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
