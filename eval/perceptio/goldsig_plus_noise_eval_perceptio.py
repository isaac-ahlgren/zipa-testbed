import argparse
import os
import sys

import numpy as np
from perceptio_tools import (
    adversary_signal,
    gen_min_events,
    generate_bits,
    golden_signal,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import Signal_Buffer, events_cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    # Setting up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-tt", "--top_threshold", type=float, default=6)
    parser.add_argument("-bt", "--bottom_threshold", type=float, default=4)
    parser.add_argument("-lt", "--lump_threshold", type=int, default=4)
    parser.add_argument("-a", "--ewma_a", type=float, default=0.75)
    parser.add_argument("-cl", "--cluster_sizes_to_check", type=int, default=4)
    parser.add_argument("-min", "--minimum_events", type=int, default=16)
    parser.add_argument("-fs", "--sampling_frequency", type=float, default=10000)
    parser.add_argument("-ch", "--chunk_size", type=int, default=100)
    parser.add_argument("-bs", "--buffer_size", type=int, default=50000)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-snr", "--snr_level", type=float, default=20)
    parser.add_argument("-t", "--trials", type=int, default=100)

    # Parsing command-line arguments
    args = parser.parse_args()
    top_th = getattr(args, "top_threshold")
    bottom_th = getattr(args, "bottom_threshold")
    lump_th = getattr(args, "lump_threshold")
    a = getattr(args, "ewma_a")
    cluster_sizes_to_check = getattr(args, "cluster_sizes_to_check")
    cluster_th = 0.1  # Set a fixed cluster threshold
    min_events = getattr(args, "minimum_events")
    Fs = getattr(args, "sampling_frequency")
    chunk_size = getattr(args, "chunk_size")
    buffer_size = getattr(args, "buffer_size")
    key_size_in_bytes = getattr(args, "key_length") // 8
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    # Generating the signals
    golden_signal = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)
    legit_signal_buffer1 = Signal_Buffer(
        golden_signal.copy(), noise=True, target_snr=target_snr
    )
    legit_signal_buffer2 = Signal_Buffer(
        golden_signal.copy(), noise=True, target_snr=target_snr
    )
    adv_signal_buffer = Signal_Buffer(adv_signal, noise=True, target_snr=target_snr)

    # Grouping the signal buffers into a tuple
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    # Defining the bit generation algorithm
    def bit_gen_algo(signal):
        signal_events, signal_event_features = gen_min_events(
            signal,
            chunk_size,
            min_events,
            top_th,
            bottom_th,
            lump_th,
            a,
        )
        bits, grouped_events = generate_bits(
            signal_events,
            signal_event_features,
            cluster_sizes_to_check,
            cluster_th,
            Fs,
            key_size_in_bytes,
        )
        return bits

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(
        events_cmp_bits, key_size_in_bytes
    )

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
