import argparse
import os
import sys

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

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import cmp_bits  # noqa: E402
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
   (window_length, band_length, key_length, target_snr, trials
    ) = get_command_line_args (
       window_length_default=16537,
    band_length_default=500,
    key_length_default=128,
    snr_level_default=20,
    trials_default=1000
    )
    

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
def bit_gen_algo(signal):
        return schurmann_wrapper_func(
            signal,
            window_length,
            band_length,
            MICROPHONE_SAMPLING_RATE,
            ANTIALIASING_FILTER,
        )

    # Creating an evaluator object with the bit generation algorithm
evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
evaluator.evaluate(signals, trials)
    # Comparing the bit errors for legitimate and adversary signals
legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)

    # Printing the average bit error rates
print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
