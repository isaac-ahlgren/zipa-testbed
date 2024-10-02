import os
import sys
from typing import List

import numpy as np
from iotcupid_tools import (
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

TOP_TH_DEFAULT = 0.07
BOTTOM_TH_DEFAULT = 0.05
LUMP_TH_DEFAULT = 4
A_DEFAULT = 0.75
CLUSTER_SIZE_TO_CHECK_DEFAULT = 4
MIN_EVENTS_DEFAULT = 16
SAMPLING_FREQ_DEFAULT = 10000
CHUNK_SIZE_DEFAULT = 10000
CLUSTER_TH_DEFAULT = 0.1
BUFFER_SIZE_DEFAULT = 50000
WINDOW_SIZE_DEFAULT = 10
FEATURE_DIM_DEFAULT = 3
QUANT_FACTOR_DEFAULT = 1
MSTART_DEFAULT = 1.1
MSTEPS_DEFAULT = 10
MEND_DEFAULT = 2
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 20
TRIALS_DEFAULT = 100


def main(
    top_th=TOP_TH_DEFAULT,
    bottom_th=BOTTOM_TH_DEFAULT,
    lump_th=LUMP_TH_DEFAULT,
    a=A_DEFAULT,
    cluster_sizes_to_check=CLUSTER_SIZE_TO_CHECK_DEFAULT,
    cluster_th=CLUSTER_TH_DEFAULT,
    min_events=MIN_EVENTS_DEFAULT,
    Fs=SAMPLING_FREQ_DEFAULT,
    chunk_size=CHUNK_SIZE_DEFAULT,
    buffer_size=BUFFER_SIZE_DEFAULT,
    window_size=WINDOW_SIZE_DEFAULT,
    feature_dimensions=FEATURE_DIM_DEFAULT,
    w=QUANT_FACTOR_DEFAULT,
    m_start=MSTART_DEFAULT,
    m_steps=MSTEPS_DEFAULT,
    m_end=MEND_DEFAULT,
    key_size_in_bytes=KEY_LENGTH_DEFAULT // 8,
    target_snr=TARGET_SNR_DEFAULT,
    trials=TRIALS_DEFAULT,
):
    mem_th = 0.8

    # Generating the signals
    signal1 = golden_signal(buffer_size)
    signal2 = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)

    signals = load_controlled_signal_buffers([signal1, signal2, adv_signal])

    # Defining the bit generation algorithm
    def bit_gen_algo(signal: Signal_Buffer) -> List[int]:
        """
        Generate cryptographic bits from the signal using IoTCupid algorithm components.

        :param signal: Signal buffer to process.
        :return: A list of generated bits.
        """
        signal_events, signal_event_signals = gen_min_events(
            signal,
            chunk_size,
            min_events,
            top_th,
            bottom_th,
            lump_th,
            a,
            window_size,
        )
        bits, grouped_events = generate_bits(
            signal_events,
            signal_event_signals,
            cluster_sizes_to_check,
            cluster_th,
            m_start,
            m_end,
            m_steps,
            w,
            feature_dimensions,
            Fs,
            key_size_in_bytes,
            mem_th,
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
        window_size_default=WINDOW_SIZE_DEFAULT,
        feature_dimensions_default=FEATURE_DIM_DEFAULT,
        quantization_factor_default=QUANT_FACTOR_DEFAULT,
        mstart_default=MSTART_DEFAULT,
        msteps_default=MSTEPS_DEFAULT,
        mend_default=MEND_DEFAULT,
        key_length_default=KEY_LENGTH_DEFAULT,
        snr_level_default=TARGET_SNR_DEFAULT,
        trials_default=TRIALS_DEFAULT,
    )
    main(*args)
