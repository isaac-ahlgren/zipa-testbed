import os
import sys
from typing import List


import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    get_command_line_args,
    golden_signal,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..") 
from evaluator import Evaluator  # noqa: E402

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
    # Calculating the number of samples needed
    sample_num = schurmann_calc_sample_num(
        key_length,
        window_length,
        band_length,
        MICROPHONE_SAMPLING_RATE,
        ANTIALIASING_FILTER,
    )

    # Generating the signals
    signal1 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signal2 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    adv_signal = adversary_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signals = (signal1, signal2, adv_signal)

    # Defining the bit generation algorithm
    def bit_gen_algo(signal: np.ndarray, *argv: List) -> np.ndarray:
        """
        Processes the signal using the Schurmann wrapper function to generate cryptographic bits.

        :param signal: The signal data to be processed.
        :type signal: np.ndarray
        :return: The processed signal data after applying the Schurmann algorithm.
        :rtype: np.ndarray
        """
        return schurmann_wrapper_func(
            signal,
            argv[0],
            argv[1],
            argv[2],
            argv[3],
        )

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate_controlled_signals(signals, trials, window_length, band_length, MICROPHONE_SAMPLING_RATE, ANTIALIASING_FILTER)
    # Comparing the bit errors for legitimate adversary signals
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
