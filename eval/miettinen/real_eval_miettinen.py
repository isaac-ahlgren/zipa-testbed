import os
import sys
from typing import List

import numpy as np
from miettinen_tools import (
    DATA_DIRECTORY,
    miettinen_wrapper_func,
    unpack_parameters,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import load_signal_groups  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File  # noqa: E402

KEY_LENGTH_DEFAULT = 128

DATA_FILE_STUB = "miettinen_real_eval_full_two_weeks"

DEFAULT_GROUPS = [
    ["10.0.0.238", "10.0.0.228", "10.0.0.239"],
    ["10.0.0.231", "10.0.0.232", "10.0.0.239"],
    ["10.0.0.233", "10.0.0.236", "10.0.0.239"],
    ["10.0.0.227", "10.0.0.229", "10.0.0.237"],
    ["10.0.0.235", "10.0.0.237", "10.0.0.239"],
    ["10.0.0.234", "10.0.0.239", "10.0.0.237"],
]

SENSOR_TYPE_DEFAULT = "mic"

TIMESTAMP_DEFAULT = "202408*"

SENSOR_DATA_DIR_DEFAULT = "/mnt/nas"

PARAM_DIR = "../plot_scripts/plot_data/miettinen_real_fuzz"
PARAM_FILE_STUB = "miettinen_real_fuzz_day1"


def main(
    key_length=KEY_LENGTH_DEFAULT,  # has to stay 128
    groups=DEFAULT_GROUPS,
    signal_data_dir=SENSOR_DATA_DIR_DEFAULT,
    parameter_data_dir=PARAM_DIR,
    parameter_file_stub=PARAM_FILE_STUB,
    sensor_type=SENSOR_TYPE_DEFAULT,
    timestamp=TIMESTAMP_DEFAULT,
):
    if not os.path.isdir(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)

    group_signals, group_params = load_signal_groups(
        groups,
        sensor_type,
        timestamp,
        signal_data_dir,
        parameter_data_dir,
        parameter_file_stub,
        unpack_parameters,
    )

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
        return output, None

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.best_parameter_evaluation(
        group_signals, group_params, key_length, DATA_DIRECTORY, DATA_FILE_STUB
    )


if __name__ == "__main__":
    main()
