import os
import random
import sys
from typing import List

import numpy as np
from fastzip_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    fastzip_event_detection_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    get_fuzzing_command_line_args,
    load_controlled_signal_files,
    log_parameters,
    make_dirs,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 2000
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
WINDOW_SIZE_RANGE = (50, 10000)

MIN_OVERLAP_DEFAULT = 0

POWER_TH_RANGE = (30, 200)

SNR_TH_RANGE = (0.5, 2)


FUZZING_DIR = "fastzip_controlled_fuzz"
FUZZING_STUB = "fastzip_controlled_event_fuzz"

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
        window_size = random.randint(
            WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1]
        )  # nosec
        overlap_size = random.randint(MIN_OVERLAP_DEFAULT, window_size // 2)  # nosec
        alpha = 0.5 #random.uniform(0, 1)  # nosec
        normalize = random.choice([True, False])  # nosec
        power_th = random.uniform(POWER_TH_RANGE[0], POWER_TH_RANGE[1])  # nosec
        snr_th = random.uniform(SNR_TH_RANGE[0], SNR_TH_RANGE[1])  # nosec
        peak_status = False
        return (
            window_size,
            overlap_size,
            power_th,
            snr_th,
            MICROPHONE_SAMPLING_RATE,
            peak_status,
            normalize,
            alpha,
        )

    def log(params, file_name_stub):
        names = ["top_th", "bottom_th", "lump_th", "a", "window_sz"]
        param_list = [params[0], params[1], params[2], params[3], params[4]]
        log_parameters(file_name_stub, names, param_list)

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        fastzip_event_detection_wrapper_func,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_gen=True,
        change_and_log_seed=True,
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