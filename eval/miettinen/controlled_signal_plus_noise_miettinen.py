import argparse
import os
import sys

import numpy as np
from miettinen_tools import miettinen_calc_sample_num, miettinen_wrapper_func

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    cmp_bits,
    load_controlled_signal,
)
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    # Setting up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--snap_shot_width", type=float, default=5)
    parser.add_argument("-f", "--no_snap_shot_width", type=float, default=5)
    parser.add_argument("-at", "--absolute_threshold", type=float, default=5e-15)
    parser.add_argument("-rt", "--relative_threshold", type=float, default=0.1)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-snr", "--snr_level", type=float, default=20)
    parser.add_argument("-t", "--trials", type=int, default=100)

    # Parsing command-line arguments
    args = parser.parse_args()
    w = getattr(args, "snap_shot_width")
    f = getattr(args, "no_snap_shot_width")
    abs_thresh = getattr(args, "absolute_threshold")
    rel_thresh = getattr(args, "relative_threshold")
    key_length = getattr(args, "key_length")
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    # Loading the controlled signals
    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )

    # Converting time durations to number of samples
    w_in_samples = int(w * sr)
    f_in_samples = int(f * sr)

    # Calculating the number of samples needed
    sample_num = miettinen_calc_sample_num(key_length, w_in_samples, f_in_samples)

    legit_signal_buffer1 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=target_snr
    )
    legit_signal_buffer2 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=target_snr
    )
    adv_signal_buffer = Signal_Buffer(adv_signal, noise=True, target_snr=target_snr)

    # Grouping the signal buffers into a tuple
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    # Defining the bit generation algorithm
    def bit_gen_algo(signal):
        signal_chunk = signal.read(sample_num)  # Reading a chunk of the signal
        return miettinen_wrapper_func(
            signal_chunk, f_in_samples, w_in_samples, rel_thresh, abs_thresh
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
