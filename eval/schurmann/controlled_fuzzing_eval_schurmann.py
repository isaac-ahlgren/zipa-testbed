import os
import sys
import random
from typing import List

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    DATA_DIRECTORY,
    get_command_line_args,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    cmp_bits,
    load_controlled_signal_files,
    log_bytes,
    log_parameters,
    gen_id
)
from evaluator import Evaluator  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
NUMBER_OF_KEYS = 100
NUMBER_OF_CHOICES_DEFAULT = 1000

# Random Parameter Ranges
WINDOW_LENGTH_RANGE = (5000, 2*48000)
MIN_BAND_LENGTH = 1

def main(
    key_length=KEY_LENGTH_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
):
    if os.path.isdir(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)

    signals = load_controlled_signal_files()

    def get_random_parameters():
        window_length = random.randint(WINDOW_LENGTH_RANGE[0], WINDOW_LENGTH_RANGE[1])
        band_length = random.randint(MIN_BAND_LENGTH, (WINDOW_LENGTH_RANGE[1] // 2 + 1) // 2)
        # Calculating the number of samples needed
        sample_num = schurmann_calc_sample_num(
            key_length, window_length, band_length, MICROPHONE_SAMPLING_RATE, ANTIALIASING_FILTER
        )
        return window_length, band_length, MICROPHONE_SAMPLING_RATE, ANTIALIASING_FILTER, sample_num

    def log(byte_list, *params):
        id = gen_id()
        os.mkdir(f"./{DATA_DIRECTORY}/schurmann_id{id}")
        file_stub = f"{DATA_DIRECTORY}/schurmann_id{id}/schurmann_id{id}"
        names = ["window_length", "band_length", "sample_num"]
        param_list = [params[0], params[1], params[4]]
        log_parameters(file_stub, names, param_list)
        log_bytes(file_stub, byte_list, key_length)

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
    evaluator = Evaluator(bit_gen_algo, random_parameter_func=get_random_parameters, logging_func=log, event_driven=False)
    evaluator.fuzzing_evaluation(signals)

if __name__ == "__main__":
    main()