import os
import sys
from typing import List

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    get_command_line_args,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import load_controlled_signal_buffers  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from signal_buffer import Signal_Buffer  # noqa: E402

WINDOW_LENGTH_DEFAULT = 16537
BAND_LENGTH_DEFAULT = 500
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 20
TRIALS_DEFAULT = 1000


def main(
    window_length=WINDOW_LENGTH_DEFAULT,
    band_length=BAND_LENGTH_DEFAULT,
    key_length=KEY_LENGTH_DEFAULT,
    target_snr=TARGET_SNR_DEFAULT,
    trials=TRIALS_DEFAULT,
):
    # Loading the controlled signals
    signals = load_controlled_signal_buffers(target_snr=target_snr, noise=True)

    # Calculating the number of samples needed
    sample_num = schurmann_calc_sample_num(
        key_length,
        window_length,
        band_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
    )

    # Defining thcontrolled_signal_fuzzinge bit generation algorithm
    def bit_gen_algo(signal: Signal_Buffer, *argv: List) -> np.ndarray:
        """
        Processes the signal using the Schurmann wrapper function to generate cryptographic bits.

        :param signal: The signal data to be processed.
        :type signal: Signal_Buffer
        :return: The processed signal data after applying the Schurmann algorithm.
        :rtype: np.ndarray
        """
        signal_chunk = signal.read(argv[4])  # Reading a chunk of the signal
        return schurmann_wrapper_func(signal_chunk, argv[0], argv[1], argv[2], argv[3])

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate_controlled_signals(
        signals,
        trials,
        window_length,
        band_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
        sample_num,
    )
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_collected_bits(key_length)

    le_avg_be = np.mean(legit_bit_errs)
    adv_avg_be = np.mean(adv_bit_errs)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {le_avg_be}")
    print(f"Adversary Average Bit Error Rate: {adv_avg_be}")
    return le_avg_be, adv_avg_be


if __name__ == "__main__":
    args = get_command_line_args(
        window_length_default=WINDOW_LENGTH_DEFAULT,
        band_length_default=BAND_LENGTH_DEFAULT,
        key_length_default=KEY_LENGTH_DEFAULT,
        snr_level_default=TARGET_SNR_DEFAULT,
        trials_default=TRIALS_DEFAULT,
    )
    main(*args)
