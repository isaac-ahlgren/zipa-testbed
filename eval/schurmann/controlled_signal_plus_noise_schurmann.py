import argparse
import os
import sys

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)
from scipy.io import wavfile

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import Signal_Buffer, add_gauss_noise, cmp_bits, load_controlled_signal  # noqa: E402
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

    # Loading the controlled signals
    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal("../../data/adversary_controlled_signal.wav")
    legit_signal_buffer1 = Signal_Buffer(legit_signal.copy(), noise=True, target_snr=target_snr)
    legit_signal_buffer2 = Signal_Buffer(legit_signal.copy(), noise=True, target_snr=target_snr)
    adv_signal_buffer = Signal_Buffer(adv_signal)

    # Grouping the signal buffers into a tuple
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    # Calculating the number of samples needed
    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, sr, ANTIALIASING_FILTER
    )

    # Defining the bit generation algorithm
    def bit_gen_algo(signal):
        signal_chunk = signal.read(sample_num)  # Reading a chunk of the signal
        return schurmann_wrapper_func(
            signal_chunk, window_length, band_length, sr, ANTIALIASING_FILTER
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