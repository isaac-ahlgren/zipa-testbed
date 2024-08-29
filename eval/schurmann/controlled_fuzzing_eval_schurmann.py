import os
import sys
import random
from typing import List

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    get_command_line_args,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    cmp_bits,
    load_controlled_signal,
    log_parameters,
)
from evaluator import Evaluator  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
NUMBER_OF_KEYS = 100
NUMBER_OF_CHOICES_DEFAULT = 1000

# Random Parameter Ranges
WINDOW_LENGTH_RANGE = (5000, 3*48000)
MIN_BAND_LENGTH = 1

def main(
    key_length=KEY_LENGTH_DEFAULT,
    number_of_keys=NUMBER_OF_KEYS,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
):
    # Loading the controlled signals
    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )
    legit_signal_buffer1 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=20
    )
    legit_signal_buffer2 = Signal_Buffer(
        legit_signal.copy(), noise=True, target_snr=20
    )
    adv_signal_buffer = Signal_Buffer(adv_signal)

    # Grouping the signal buffers into a tuple
    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    def get_random_parameters():
        window_length = random.randint(WINDOW_LENGTH_RANGE[0], WINDOW_LENGTH_RANGE[1])
        band_length = random.randint(MIN_BAND_LENGTH, (WINDOW_LENGTH_RANGE[1] // 2 + 1) // 2)
        # Calculating the number of samples needed
        sample_num = schurmann_calc_sample_num(
            key_length, window_length, band_length, sr, ANTIALIASING_FILTER
        )
        return window_length, band_length, sr, ANTIALIASING_FILTER, sample_num

    base_file_name = "schurmann_ber_id"
    def log(legit_bit_errs, adv_bit_errs, *params):
        names = ["window_length", "band_length", "sample_num"]
        param_list = [params[0], params[1], params[4]]
        id = random.randint(0, 2**64)
        file_name = base_file_name + str(id)
        log_parameters(file_name, names, param_list, legit_bit_errs, adv_bit_errs)

    def bit_gen_algo(signal: Signal_Buffer, *argv: List) -> np.ndarray:
        """
        Processes the signal using the Schurmann wrapper function to generate cryptographic bits.

        :param signal: The signal data to be processed.
        :type signal: Signal_Buffer
        :return: The processed signal data after applying the Schurmann algorithm.
        :rtype: np.ndarray
        """
        signal_chunk = signal.read(argv[4])  # Reading a chunk of the signal
        return schurmann_wrapper_func(
            signal_chunk, argv[0], argv[1], argv[2], argv[3]
        ) 
    
    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    evaluator.fuzzing_evaluation(signals, log, get_random_parameters, cmp_bits, number_of_keys, number_of_choices, key_length)

if __name__ == "__main__":
    main()