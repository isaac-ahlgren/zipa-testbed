import os
import random
import sys
from typing import List

import numpy as np
from miettinen_tools import (
    DATA_DIRECTORY,
    miettinen_calc_sample_num,
    miettinen_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    load_real_signal_files,
    log_parameters,
    make_dirs,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
W_LENGTH_RANGE = (5000, 10 * 48000)
F_LENGTH_RANGE = (5000, 10 * 48000)
REL_THR_RANGE = (-1, 15)
ABS_THR_RANGE = (0, 200000)

FUZZING_DIR = "miettinen_real_fuzz"
FUZZING_STUB = "miettinen_real_fuzz_day1"

DEFAULT_IDS = [
    "10.0.0.238",
    "10.0.0.228",
    "10.0.0.231",
    "10.0.0.232",
    "10.0.0.233",
    "10.0.0.236",
    "10.0.0.227",
    "10.0.0.229",
    "10.0.0.235",
    "10.0.0.237",
    "10.0.0.234",
    "10.0.0.239",
]

DEFAULT_SENSOR_TYPE = "mic"

DEFAULT_TIMESTAMP = "20240813*"

SENSOR_DATA_DIR = "/mnt/nas"


def main(
    key_length=KEY_LENGTH_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
    data_dir=SENSOR_DATA_DIR,
    sensor_type=DEFAULT_SENSOR_TYPE,
    dev_ids=DEFAULT_IDS,
    timestamp=DEFAULT_TIMESTAMP,
):
    make_dirs(DATA_DIRECTORY, FUZZING_DIR, FUZZING_STUB)

    fuzzing_dir = f"{DATA_DIRECTORY}/{FUZZING_DIR}/{FUZZING_STUB}"

    signals = load_real_signal_files(data_dir, dev_ids, sensor_type, timestamp)

    def get_random_parameters():
        w_length = random.randint(W_LENGTH_RANGE[0], W_LENGTH_RANGE[1])  # nosec
        f_length = random.randint(F_LENGTH_RANGE[0], F_LENGTH_RANGE[1])  # nosec
        rel_thr = random.uniform(REL_THR_RANGE[0], REL_THR_RANGE[1])  # nosec
        abs_thr = random.uniform(ABS_THR_RANGE[0], ABS_THR_RANGE[1])  # nosec
        # Calculating the number of samples needed
        sample_num = miettinen_calc_sample_num(
            key_length,
            w_length,
            f_length,
        )
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
        return output, None

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        bit_gen_algo,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_gen=False,
    )
    evaluator.fuzzing_evaluation(
        signals,
        number_of_choices,
        key_length,
        fuzzing_dir,
        FUZZING_STUB,
        multithreaded=True,
    )


if __name__ == "__main__":
    main()
