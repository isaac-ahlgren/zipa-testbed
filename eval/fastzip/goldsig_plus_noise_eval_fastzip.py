import os
import sys
from typing import ByteString

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
from eval_tools import load_controlled_signal_buffers  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_Buffer  # noqa: E402

WINDOW_SIZE_DEFAULT = 200
OVERLAP_SIZE_DEFAULT = 100
BUFFER_SIZE_DEAFULT = 50000
N_BITS_DEFAULT = 12
KEY_LENGTH_DEFAULT = 128
BIAS_DEFAULT = 0
EQD_DELTA_DEFAULT = 1
PEAK_STATUS_DEFAULT = None
EWMA_FILTER_DEFAULT = None
ALPHA_DEFAULT = None
REMOVE_NOISE_DEFAULT = None
NORMALIZE_DEFAULT = True
POWER_TH_DEFAULT = -12
SNR_TH_DEFAULT = 1.6
NUM_PEAKS_DEFAULT = 0
TARGET_SNR_DEFAULT = 20
TRIALS_DEFAULT = 1000


def main(
    window_size=WINDOW_SIZE_DEFAULT,
    overlap_size=OVERLAP_SIZE_DEFAULT,
    buffer_size=BUFFER_SIZE_DEAFULT,
    n_bits=N_BITS_DEFAULT,
    key_length=KEY_LENGTH_DEFAULT,
    bias=BIAS_DEFAULT,
    eqd_delta=EQD_DELTA_DEFAULT,
    peak_status=PEAK_STATUS_DEFAULT,
    ewma_filter=EWMA_FILTER_DEFAULT,
    alpha=ALPHA_DEFAULT,
    remove_noise=REMOVE_NOISE_DEFAULT,
    normalize=NORMALIZE_DEFAULT,
    power_threshold=POWER_TH_DEFAULT,
    snr_threshold=SNR_TH_DEFAULT,
    number_peaks=NUM_PEAKS_DEFAULT,
    target_snr=TARGET_SNR_DEFAULT,
    trials=TRIALS_DEFAULT,
):
    signal1 = golden_signal(buffer_size)
    signal2 = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)
    signals = load_controlled_signal_buffers(
        [signal1, signal2, adv_signal], target_snr=target_snr, noise=True
    )

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
    evaluator.evaluate_controlled_signals(signals, trials)

    legit_bit_errs, adv_bit_errs = evaluator.cmp_collected_bits(key_length)

    le_avg_be = np.mean(legit_bit_errs)
    adv_avg_be = np.mean(adv_bit_errs)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {le_avg_be}")
    print(f"Adversary Average Bit Error Rate: {adv_avg_be}")
    return le_avg_be, adv_avg_be


if __name__ == "__main__":
    args = parse_command_line_args(
        window_size_default=WINDOW_SIZE_DEFAULT,
        overlap_size_default=OVERLAP_SIZE_DEFAULT,
        buffer_size_default=BUFFER_SIZE_DEAFULT,
        n_bits_default=N_BITS_DEFAULT,
        key_length_default=KEY_LENGTH_DEFAULT,
        bias_default=BIAS_DEFAULT,
        eqd_delta_default=EQD_DELTA_DEFAULT,
        peak_status_default=PEAK_STATUS_DEFAULT,
        ewma_filter_default=EWMA_FILTER_DEFAULT,
        alpha_default=ALPHA_DEFAULT,
        remove_noise_default=REMOVE_NOISE_DEFAULT,
        normalize_default=NORMALIZE_DEFAULT,
        power_threshold_default=POWER_TH_DEFAULT,
        snr_threshold_default=SNR_TH_DEFAULT,
        number_peaks_default=NUM_PEAKS_DEFAULT,
        target_snr_default=TARGET_SNR_DEFAULT,
        trials_default=TRIALS_DEFAULT,
    )
    main(*args)
