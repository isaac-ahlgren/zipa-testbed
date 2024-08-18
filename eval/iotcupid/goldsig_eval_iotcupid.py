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
from eval_tools import Signal_Buffer, events_cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402


def main(
    top_th,
    bottom_th,
    lump_th,
    a,
    cluster_sizes_to_check,
    cluster_th,
    min_events,
    Fs,
    chunk_size,
    buffer_size,
    window_size,
    feature_dimensions,
    w,
    m_start,
    m_steps,
    m_end,
    key_size_in_bytes,
    target_snr,
    trials,
):

    mem_th = 0.8

    # Generating the signals
    gold_signal = golden_signal(buffer_size)
    adv_signal = adversary_signal(buffer_size)
    legit_signal_buffer1 = Signal_Buffer(gold_signal.copy())
    legit_signal_buffer2 = Signal_Buffer(gold_signal.copy())
    adv_signal_buffer = Signal_Buffer(adv_signal)

    # Grouping the signal buffers into a tuple
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

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
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(
        events_cmp_bits, key_size_in_bytes
    )

    le_avg_be = np.mean(legit_bit_errs)
    adv_avg_be = np.mean(adv_bit_errs)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {le_avg_be}")
    print(f"Adversary Average Bit Error Rate: {adv_avg_be}")
    return le_avg_be, adv_avg_be


if __name__ == "__main__":
    args = get_command_line_args(
        top_threshold_default=0.07,
        bottom_threshold_default=0.05,
        lump_threshold_default=4,
        ewma_a_default=0.75,
        cluster_sizes_to_check_default=4,
        minimum_events_default=16,
        sampling_frequency_default=10000,
        chunk_size_default=10000,
        buffer_size_default=50000,
        window_size_default=10,
        feature_dimensions_default=3,
        quantization_factor_default=1,
        mstart_default=1.1,
        msteps_default=10,
        mend_default=2,
        key_length_default=128,
        snr_level_default=1,
        trials_default=100,
    )
    main(*args)
