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
    parse_command_line_args,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import Signal_Buffer, cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    (
        window_size, overlap_size, buffer_size, n_bits, key_length, bias, eqd_delta,
        peak_status, ewma_filter, alpha, remove_noise, normalize, power_threshold,
        snr_threshold, number_peaks, snr_level, trials,target_snr
    ) = parse_command_line_args(
        window_size_default=200,
        overlap_size_default=100,
        buffer_size_default=50000,
        n_bits_default=12,
        key_length_default=128,
        bias_default=0,
        eqd_delta_default=1,
        peak_status_default=None,
        ewma_filter_default=None,
        alpha_default=None,
        remove_noise_default=None,
        normalize_default=True,
        power_threshold_default=-12,
        snr_threshold_default=1.6,
        number_peaks_default=0,
        snr_level_default=20,
        trials_default=1000

    )
    signal1 = golden_signal(buffer_size) 
    signal2 = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)
    legit_signal_buffer1 = Signal_Buffer(signal1, noise=True, target_snr=target_snr)
    legit_signal_buffer2 = Signal_Buffer(signal2, noise=True, target_snr=target_snr)
    adv_signal_buffer = Signal_Buffer(adv_signal, noise=True, target_snr=target_snr)
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
