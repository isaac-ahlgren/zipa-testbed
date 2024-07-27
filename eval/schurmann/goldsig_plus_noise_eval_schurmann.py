import argparse
import os
import sys

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    golden_signal,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import add_gauss_noise, cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    # Setting up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-wl", "--window_length", type=int, default=16537)
    parser.add_argument("-bl", "--band_length", type=int, default=500)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-snr", "--snr_level", type=float, default=20)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    # Parsing command-line arguments
    args = parser.parse_args()
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    # Calculating the number of samples needed
    sample_num = schurmann_calc_sample_num(
        key_length,
        window_length,
        band_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
    )

    # Generating the signals
    signal1 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signal2 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    adv_signal = adversary_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signals = (signal1, signal2, adv_signal)

    # Defining the bit generation algorithm
    def bit_gen_algo(signal):
        noisy_signal = add_gauss_noise(signal, target_snr)
        return schurmann_wrapper_func(
            noisy_signal,
            window_length,
            band_length,
            MICROPHONE_SAMPLING_RATE,
            ANTIALIASING_FILTER,
        )

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
