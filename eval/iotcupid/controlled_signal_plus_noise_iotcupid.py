import argparse
import os
import sys

import numpy as np
from iotcupid_tools import (
    gen_min_events,
    generate_bits,
    get_command_line_args
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    events_cmp_bits,
    load_controlled_signal,
)
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
  
    get_command_line_args(
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
        trials_default=100
    )
    if __name__ == "__main__":

        args = get_command_line_args()
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
    def bit_gen_algo(signal):
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

