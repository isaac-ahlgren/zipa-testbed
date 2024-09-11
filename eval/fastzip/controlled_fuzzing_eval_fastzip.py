import os
import random
import sys
from typing import List

import numpy as np
from fastzip_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    fastzip_wrapper_function,
    manage_overlapping_chunks,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    make_dirs,
    get_fuzzing_command_line_args,
    load_controlled_signal_files,
    log_parameters,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
WINDOW_SIZE_RANGE = (50, 10000)

MIN_OVERLAP_DEFAULT = 0
OVERLAP_SIZE_DEFAULT = () # max val half of the chosen window size

MIN_N_BITS_DEFAULT = 10
N_BITS_DEFAULT = 18 # max half of chosen window size 

BIAS_DEFAULT = 0 # no need to worry

EQD_DELTA_DEFAULT = 1 # dependent of window_size /  n_bits (smaller)
PEAK_STATUS_DEFAULT = None
EWMA_FILTER_DEFAULT = None
ALPHA_DEFAULT = None # between 0 and 1
REMOVE_NOISE_DEFAULT = None
NORMALIZE_DEFAULT = True

POWER_TH_RANGE = (30, 200)

SNR_TH_RANGE = (0.5, 2)

# no need to worry
PEAK_STATUS_DEFAULT = None
NUM_PEAKS_DEFAULT = 0
BIAS_DEFAULT = 0

FUZZING_DIR = "fastzip_controlled_fuzz"
FUZZING_STUB = "fastzip_controlled_fuzz"

def main(
    key_length=KEY_LENGTH_DEFAULT,
    target_snr = TARGET_SNR_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
    wrap_around_limit=WRAP_AROUND_LIMIT_DEFAULT
):
    make_dirs(DATA_DIRECTORY, FUZZING_DIR, f"{FUZZING_STUB}_snr{target_snr}")

    fuzzing_dir = f"{DATA_DIRECTORY}/{FUZZING_DIR}/{FUZZING_STUB}_snr{target_snr}"

    signals = load_controlled_signal_files(target_snr, wrap_around=True, wrap_around_limit=wrap_around_limit)

    def get_random_parameters():
        window_size = random.randint(
            WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1]
        )  # nosec
        overlap_size = random.randint(
            MIN_OVERLAP_DEFAULT, window_size // 2
        )  # nosec
        rel_thr = random.uniform(REL_THR_RANGE[0], REL_THR_RANGE[1]
        )  # nosec
        abs_thr = random.uniform(ABS_THR_RANGE[0], ABS_THR_RANGE[1]
        )  # nosec
        return (
            f_length,
            w_length,
            rel_thr,
            abs_thr,
            sample_num,
        )

    def log(params, file_name_stub):
        names = ["f_samples", "w_samples", "rel_thr", "abs_thr", "sample_num"]
        param_list = [params[0], params[1], params[2], params[3], params[4]]
        log_parameters(file_name_stub, names, param_list)

    def bit_gen_algo(signal: Signal_File, *argv: List) -> np.ndarray:
        """
        Processes the signal using the Schurmann wrapper function to generate cryptographic bits.

        :param signal: The signal data to be processed.
        :type signal: Signal_Buffer
        :return: The processed signal data after applying the Schurmann algorithm.
        :rtype: np.ndarray
        """
        read_length = argv[4]
        signal_chunk = signal.read(argv[4])  # Reading a chunk of the signal
        if len(signal_chunk) == read_length:
            output = miettinen_wrapper_func(
                signal_chunk, argv[0], argv[1], argv[2], argv[3]
            )
        else:
            output = None
        return output

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        bit_gen_algo,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_driven=False,
    )
    evaluator.fuzzing_evaluation(signals, number_of_choices, key_length, fuzzing_dir, f"{FUZZING_STUB}_snr{target_snr}", multithreaded=True)


if __name__ == "__main__":
    args = get_fuzzing_command_line_args(key_length_default = KEY_LENGTH_DEFAULT,
                                         target_snr_default = TARGET_SNR_DEFAULT,
                                         number_of_choices_default = NUMBER_OF_CHOICES_DEFAULT,
                                         wrap_around_limit_default = WRAP_AROUND_LIMIT_DEFAULT,)
    main(*args)