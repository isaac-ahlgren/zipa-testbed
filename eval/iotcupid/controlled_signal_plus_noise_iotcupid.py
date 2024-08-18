import os
import sys
from typing import List

import numpy as np
from iotcupid_tools import gen_min_events, generate_bits, get_command_line_args

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    events_cmp_bits,
    load_controlled_signal,
)
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

    # Loading the controlled signals
    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )

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
    def bit_gen_algo(signal: Signal_Buffer) -> List[int]:
        """
        Generate cryptographic bits from an input signal using the defined Perceptio processing algorithm.

        This function orchestrates the generation of cryptographic bits by first detecting events within the signal and then processing these events to generate bits. The process involves event detection with specific thresholds and conditions, followed by a clustering process that organizes these events into meaningful groups from which cryptographic bits are derived.

        :param signal: The signal data to be processed, encapsulated in a `Signal_Buffer` object which provides an interface for reading signal chunks.
        :type signal: Signal_Buffer
        :return: A list of integers representing the cryptographic bits generated from the signal.
        :rtype: List[int]
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
        trials_default=10,
    )
    main(*args)
