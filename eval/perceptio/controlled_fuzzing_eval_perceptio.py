import os
import random
import sys
from typing import List

import numpy as np
from perceptio_tools import (
    DATA_DIRECTORY,
    process_events,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    get_fuzzing_command_line_args,
    load_controlled_signal_files,
    log_parameters,
    make_dirs,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File, Event_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
CLUSTER_SZ_RANGE = (1, 5)
CLUSTER_TH_RANGE = (0.1, 0.5)

FUZZING_DIR = "perceptio_controlled_fuzz"
FUZZING_STUB = "perceptio_controlled_fuzz"


def main(
    key_length=KEY_LENGTH_DEFAULT,
    target_snr=TARGET_SNR_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
    wrap_around_limit=WRAP_AROUND_LIMIT_DEFAULT,
):
    make_dirs(DATA_DIRECTORY, FUZZING_DIR, f"{FUZZING_STUB}_snr{target_snr}")

    fuzzing_dir = f"{DATA_DIRECTORY}/{FUZZING_DIR}/{FUZZING_STUB}_snr{target_snr}"

    signals = load_controlled_signal_files(
        target_snr, wrap_around=True, wrap_around_limit=wrap_around_limit
    )

    def get_random_parameters():
        cluster_size = random.randint(CLUSTER_SZ_RANGE[0], CLUSTER_SZ_RANGE[1])  # nosec
        cluster_th = random.uniform(CLUSTER_TH_RANGE[0], CLUSTER_TH_RANGE[1])  # nosec
        # Calculating the number of samples needed
        return (
            cluster_size,
            cluster_th,
        )

    def log(params, file_name_stub):
        names = ["top_th", "bottom_th", "lump_th", "a", "cluster_size", "cluster_th"]
        param_list = [params[0], params[1], params[2], params[3], params[4], params[5]]
        log_parameters(file_name_stub, names, param_list)

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        process_events,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_gen=True,
    )
    evaluator.fuzzing_evaluation(
        signals,
        number_of_choices,
        key_length,
        fuzzing_dir,
        f"{FUZZING_STUB}_snr{target_snr}",
        multithreaded=True,
    )

if __name__ == "__main__":
    main()