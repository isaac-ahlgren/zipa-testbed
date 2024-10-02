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
    load_real_signal_files,
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
 
ALPHA_DEFAULT = None  # between 0 and 1
NORMALIZE_DEFAULT = True

POWER_TH_RANGE = (30, 200)

SNR_TH_RANGE = (0.5, 10)

FUZZING_DIR = "fastzip_event_real_fuzz"
FUZZING_STUB = "fastzip_event_real_fuzz_day1"

DEFAULT_IDS = ["10.0.0.238","10.0.0.228",
            "10.0.0.231","10.0.0.232",
            "10.0.0.233","10.0.0.236",
            "10.0.0.227","10.0.0.229",
            "10.0.0.235","10.0.0.237",
            "10.0.0.234","10.0.0.239"]

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
        window_size = random.randint(
            WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1]
        )  # nosec
        overlap_size = random.randint(MIN_OVERLAP_DEFAULT, window_size // 2)  # nosec
        alpha = None #random.uniform(0, 1)  # nosec
        normalize = random.choice([True, False])  # nosec
        power_th = random.uniform(POWER_TH_RANGE[0], POWER_TH_RANGE[1])  # nosec
        snr_th = random.uniform(SNR_TH_RANGE[0], SNR_TH_RANGE[1])  # nosec
        peak_th = 0
        peak_status = False
        return (
            window_size,
            overlap_size,
            power_th,
            snr_th,
            peak_th,
            MICROPHONE_SAMPLING_RATE,
            peak_status,
            normalize,
            alpha,
        )

    def log(params, file_name_stub):
        names = [
            "window_size",
            "overlap_size",
            "power_th",
            "snr_thr",
            "peak_th",
            "sampling_rate",
            "peak_status",
            "normalize",
            "alpha",
        ]
        param_list = [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
        ]
        log_parameters(file_name_stub, names, param_list)


    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        fastzip_event_detection_wrapper_func,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_gen=True,
    )

    evaluator.fuzzing_evaluation(
        signals,
        number_of_choices,
        key_length,
        fuzzing_dir,
        f"{FUZZING_STUB}",
        multithreaded=True,
    )

if __name__ == "__main__":
    main()

