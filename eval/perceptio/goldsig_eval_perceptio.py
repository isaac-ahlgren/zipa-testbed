import os
import sys
from typing import ByteString

import numpy as np
from perceptio_tools import (
    adversary_signal,
    gen_min_events,
    generate_bits,
    get_command_line_args,
    golden_signal,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import load_controlled_signal_buffers  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_Buffer  # noqa: E402

TOP_TH_DEFAULT = 6
BOTTOM_TH_DEFAULT = 4
LUMP_TH_DEFAULT = 4
A_DEFAULT = 0.75
CLUSTER_SIZE_TO_CHECK_DEFAULT = 4
CLUSTER_TH_DEFAULT = 0.1
MIN_EVENTS_DEFAULT = 16
SAMPLING_FREQ_DEFAULT = 10000
CHUNK_SIZE_DEFAULT = 10000
BUFFER_SIZE_DEFAULT = 50000
KEY_SIZE_DEFAULT = 128
TARGET_SNR_DEFAULT = 20
TRIALS_DEFAULT = 100


def main(
    top_th=TOP_TH_DEFAULT,
    bottom_th=BOTTOM_TH_DEFAULT,
    lump_th=LUMP_TH_DEFAULT,
    a=A_DEFAULT,
    cluster_sizes_to_check=CLUSTER_SIZE_TO_CHECK_DEFAULT,
    cluster_th=CLUSTER_SIZE_TO_CHECK_DEFAULT,
    min_events=MIN_EVENTS_DEFAULT,
    Fs=SAMPLING_FREQ_DEFAULT,
    chunk_size=CHUNK_SIZE_DEFAULT,
    buffer_size=BUFFER_SIZE_DEFAULT,
    key_size_in_bytes=KEY_SIZE_DEFAULT // 8,
    target_snr=TARGET_SNR_DEFAULT,
    trials=TRIALS_DEFAULT,
):

    # Generating the signals
    signal1 = golden_signal(buffer_size)
    signal2 = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)

    signals = load_controlled_signal_buffers([signal1, signal2, adv_signal])

    # Defining the bit generation algorithm
    def bit_gen_algo(signal: Signal_Buffer) -> ByteString:
        """
        Generates bits based on the analysis of overlapping chunks from a signal.

        :param signal: The signal buffer to process.
        :type signal: Signal_Buffer
        :return: A byte string of the generated bits up to the specified key length.
        :rtype: ByteString
        """
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
    evaluator = Evaluator(bit_gen_algo, event_gen=True)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate_controlled_signals(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_collected_bits(key_size_in_bytes * 8)

    le_avg_be = np.mean(legit_bit_errs)
    adv_avg_be = np.mean(adv_bit_errs)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {le_avg_be}")
    print(f"Adversary Average Bit Error Rate: {adv_avg_be}")
    return le_avg_be, adv_avg_be


if __name__ == "__main__":
    args = get_command_line_args(
        top_threshold_default=TOP_TH_DEFAULT,
        bottom_threshold_default=BOTTOM_TH_DEFAULT,
        lump_threshold_default=LUMP_TH_DEFAULT,
        ewma_a_default=A_DEFAULT,
        cluster_sizes_to_check_default=CLUSTER_SIZE_TO_CHECK_DEFAULT,
        minimum_events_default=MIN_EVENTS_DEFAULT,
        sampling_frequency_default=SAMPLING_FREQ_DEFAULT,
        chunk_size_default=CHUNK_SIZE_DEFAULT,
        buffer_size_default=BUFFER_SIZE_DEFAULT,
        key_length_default=KEY_SIZE_DEFAULT,
        snr_level_default=TARGET_SNR_DEFAULT,
        trials_default=TRIALS_DEFAULT,
    )
    main(*args)
