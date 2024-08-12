import os
import sys
from typing import ByteString

import numpy as np
from fastzip_tools import (
    fastzip_wrapper_function,
    manage_overlapping_chunks,
    parse_command_line_args,
)

# from typing import List, Optional, Tuple


sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    cmp_bits,
    load_controlled_signal,
)
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    (
        window_size,
        overlap_size,
        buffer_size,
        n_bits,
        key_length,
        bias,
        eqd_delta,
        peak_status,
        ewma_filter,
        alpha,
        remove_noise,
        normalize,
        power_threshold,
        snr_threshold,
        number_peaks,
        snr_level,
        trials,
    ) = parse_command_line_args(
        window_size_default=200,
        overlap_size_default=100,
        buffer_size_default=50000,
        n_bits_default=18,
        key_length_default=128,
        bias_default=0,
        eqd_delta_default=1,
        peak_status_default=None,
        ewma_filter_default=None,
        alpha_default=None,
        remove_noise_default=None,
        normalize_default=True,
        power_threshold_default=70,
        snr_threshold_default=1.2,
        number_peaks_default=0,
        snr_level_default=20,
        trials_default=1000,
    )

    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )

    legit_signal_buffer1 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=snr_level
    )
    legit_signal_buffer2 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=snr_level
    )
    adv_signal_buffer = Signal_Buffer(adv_signal, noise=True, target_snr=snr_level)

    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    def bit_gen_algo(signal: Signal_Buffer) -> ByteString:
        """
        Generates bits based on the analysis of overlapping chunks from a signal.

        :param signal: The signal buffer to process.
        :type signal: Signal_Buffer
        :return: A byte string of the generated bits up to the specified key length.
        :rtype: ByteString
        """
        accumulated_bits = b""
        for chunk in manage_overlapping_chunks(signal, window_size, overlap_size):
            bits = fastzip_wrapper_function(
                chunk,
                n_bits,
                power_threshold,
                snr_threshold,
                number_peaks,
                bias,
                sr,
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
