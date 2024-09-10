import os
import random
import sys
import argparse
from typing import List

import numpy as np
from schurmann_tools import (
    ANTIALIASING_FILTER,
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    load_controlled_signal_files,
    log_parameters,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 10
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
WINDOW_LENGTH_RANGE = (5000, 2 * 48000)
MIN_BAND_LENGTH = 1

SCHURMANN_CONTROLLED_FUZZING = "schurmann_controlled_fuzz"
SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_controlled_fuzz"

def get_command_line_args(
    key_length_default: int = KEY_LENGTH_DEFAULT,
    target_snr_default: int = TARGET_SNR_DEFAULT,
    number_of_choices_default: int = NUMBER_OF_CHOICES_DEFAULT,
    wrap_around_limit_default: float = WRAP_AROUND_LIMIT_DEFAULT,
):
    """
    Parse command-line arguments for the script.

    :return: Tuple containing window length, band length, key length, SNR level, and number of trials.
    """
    parser = argparse.ArgumentParser()

    # Add arguments without descriptions
    parser.add_argument("-kl", "--key_length", type=int, default=key_length_default)
    parser.add_argument("-snr", "--snr_level", type=float, default=target_snr_default)
    parser.add_argument("-c", "--choices", type=int, default=number_of_choices_default)
    parser.add_argument("-wwl", "--wrap_around_limit", type=int, default=wrap_around_limit_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    key_length = getattr(args, "key_length")
    target_snr = getattr(args, "snr_level")
    number_of_choices = getattr(args, "choices")
    wrap_around_limit = getattr(args, "wrap_around_limit")

    return key_length, target_snr, number_of_choices, wrap_around_limit


def main(
    key_length=KEY_LENGTH_DEFAULT,
    target_snr = TARGET_SNR_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
    wrap_around_limit=WRAP_AROUND_LIMIT_DEFAULT
):
    if not os.path.isdir(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)

    if not os.path.isdir(f"{DATA_DIRECTORY}/{SCHURMANN_CONTROLLED_FUZZING}"):
        os.mkdir(f"{DATA_DIRECTORY}/{SCHURMANN_CONTROLLED_FUZZING}")

    if not os.path.isdir(f"{DATA_DIRECTORY}/{SCHURMANN_CONTROLLED_FUZZING}/{SCHURMANN_CONTROLLED_FUZZING}_snr{target_snr}"):
        os.mkdir(f"{DATA_DIRECTORY}/{SCHURMANN_CONTROLLED_FUZZING}/{SCHURMANN_CONTROLLED_FUZZING}_snr{target_snr}")

    fuzzing_dir = f"{DATA_DIRECTORY}/{SCHURMANN_CONTROLLED_FUZZING}/{SCHURMANN_CONTROLLED_FUZZING}_snr{target_snr}"

    signals = load_controlled_signal_files(target_snr, wrap_around=True, wrap_around_limit=wrap_around_limit)

    def get_random_parameters():
        window_length = random.randint(
            WINDOW_LENGTH_RANGE[0], WINDOW_LENGTH_RANGE[1]
        )  # nosec
        band_length = random.randint(
            MIN_BAND_LENGTH, (window_length // 2 + 1) // 2
        )  # nosec
        # Calculating the number of samples needed
        sample_num = schurmann_calc_sample_num(
            key_length,
            window_length,
            band_length,
            MICROPHONE_SAMPLING_RATE,
            ANTIALIASING_FILTER,
        )
        return (
            window_length,
            band_length,
            MICROPHONE_SAMPLING_RATE,
            ANTIALIASING_FILTER,
            sample_num,
        )

    def log(params, file_name_stub):
        names = ["window_length", "band_length", "sample_num"]
        param_list = [params[0], params[1], params[4]]
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
            output = schurmann_wrapper_func(
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
    evaluator.fuzzing_evaluation(signals, number_of_choices, key_length, fuzzing_dir, f"{SCHURMANN_CONTROLLED_FUZZING_STUB}_snr{target_snr}", multithreaded=True)


if __name__ == "__main__":
    args = get_command_line_args()
    main(*args)
