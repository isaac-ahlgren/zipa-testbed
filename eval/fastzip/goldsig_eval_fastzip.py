import argparse
import os
import sys

import numpy as np
from fastzip_tools import (
    SAMPLING_RATE,
    adversary_signal,
    fastzip_wrapper_function,
    golden_signal,
    manage_overlapping_chunks,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import Signal_Buffer, cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ws", "--window_size", type=int, default=200)
    parser.add_argument("-os", "--overlap_size", type=int, default=100)
    parser.add_argument("-bs", "--buffer_size", type=int, default=50000)
    parser.add_argument("-nb", "--n_bits", type=int, default=12)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-b", "--bias", type=int, default=0)
    parser.add_argument("-ed", "--eqd_delta", type=int, default=1)
    parser.add_argument("-ef", "--ewma_filter", type=bool, default=None)
    parser.add_argument("-ps", "--peak_status", type=bool, default=None)
    parser.add_argument("-a", "--alpha", type=float, default=None)
    parser.add_argument("-rn", "--remove_noise", type=bool, default=None)
    parser.add_argument("-n", "--normalize", type=bool, default=True)
    parser.add_argument("-pt", "--power_threshold", type=int, default=-12)
    parser.add_argument("-st", "--snr_threshold", type=int, default=1.6)
    parser.add_argument("-np", "--number_peaks", type=int, default=0)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    args = parser.parse_args()
    window_size = getattr(args, "window_size")
    overlap_size = getattr(args, "overlap_size")
    buffer_size = getattr(args, "buffer_size")
    n_bits = getattr(args, "n_bits")
    key_length = getattr(args, "key_length")
    bias = getattr(args, "bias")
    eqd_delta = getattr(args, "eqd_delta")
    peak_status = getattr(args, "peak_status")
    ewma_filter = getattr(args, "ewma_filter")
    alpha = getattr(args, "alpha")
    remove_noise = getattr(args, "remove_noise")
    normalize = getattr(args, "normalize")
    power_threshold = getattr(args, "power_threshold")
    snr_threshold = getattr(args, "snr_threshold")
    number_peaks = getattr(args, "number_peaks")
    trials = getattr(args, "trials")

    signal1 = golden_signal(buffer_size)
    signal2 = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)

    legit_signal_buffer1 = Signal_Buffer(signal1)
    legit_signal_buffer2 = Signal_Buffer(signal2)
    adv_signal_buffer = Signal_Buffer(adv_signal)
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    def bit_gen_algo(signal):
        accumulated_bits = b""
        for chunk in manage_overlapping_chunks(signal, window_size, overlap_size):
            bits = fastzip_wrapper_function(
                chunk,
                n_bits,
                power_threshold,
                snr_threshold,
                number_peaks,
                bias,
                SAMPLING_RATE,
                eqd_delta,
                peak_status,
                ewma_filter,
                alpha,
                remove_noise,
                normalize,
            )

            if bits:
                accumulated_bits += bits
                if len(accumulated_bits) >= key_length:
                    break
        if len(accumulated_bits) > key_length:
            return accumulated_bits[:key_length]
        return accumulated_bits

    evaluator = Evaluator(bit_gen_algo)
    evaluator.evaluate(signals, trials)

    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)

    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
