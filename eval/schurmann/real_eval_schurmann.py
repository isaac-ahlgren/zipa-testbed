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
from eval_tools import load_real_signal_files, log_bytes  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_Buffer  # noqa: E402

WINDOW_LENGTH_DEFAULT = 461598
BAND_LENGTH_DEFAULT = 21460
KEY_LENGTH_DEFAULT = 128

FUZZING_DIR = "schurmann_real_fuzz"
FUZZING_STUB = "schurmann_real_fuzz_full_two_weeks"

DEFAULT_IDS = ["10.0.0.238","10.0.0.228",
            "10.0.0.231","10.0.0.232",
            "10.0.0.233","10.0.0.236",
            "10.0.0.227","10.0.0.229",
            "10.0.0.235","10.0.0.237",
            "10.0.0.234","10.0.0.239"]

DEFAULT_GROUPS = [("10.0.0.238", "10.0.0.228", "10.0.0.239"),
                  ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
                  ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
                  ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
                  ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
                  ("10.0.0.234", "10.0.0.239", "10.0.0.237")]

SENSOR_TYPE_DEFAULT = "mic"

TIMESTAMP_DEFAULT = "202408*"

SENSOR_DATA_DIR_DEFAULT = "/mnt/nas"

# TODO: Write function to arrange signal files into groups and load in best set of parameters
def arrange_signals_and_params(signals, groups, best_param_dir):
    group_signals = []
    group_parameters = []
    for group in groups:
        id1 = group[0]
        id2 = group[1]
        id3 = group[2]
        

def main(
    window_length=WINDOW_LENGTH_DEFAULT,
    band_length=BAND_LENGTH_DEFAULT,
    key_length=KEY_LENGTH_DEFAULT,
    legit1_id=LEGIT1_DEFAULT,
    legit2_id=LEGIT2_DEFAULT,
    adv_id=ADV_DEFAULT,
    data_dir=SENSOR_DATA_DIR_DEFAULT,
    sensor_type=SENSOR_TYPE_DEFAULT,
    timestamp=TIMESTAMP_DEFAULT
):
    dev_ids = [legit1_id, legit2_id, adv_id]
    signals = load_real_signal_files(data_dir, dev_ids, sensor_type, timestamp)

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
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.best_parameter_evaluation(self, group_signals, group_params, key_length, dir, file_stub)

if __name__ == "__main__":
    main()
