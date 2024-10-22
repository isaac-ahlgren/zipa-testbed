import os
import random
import sys
from typing import List

import numpy as np
from perceptio_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    process_events,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    calc_all_event_bits,
    get_fuzzing_command_line_args,
    load_random_events,
    load_real_signal_groups,
    log_parameters,
    make_dirs,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Event_File, Signal_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
NUMBER_OF_CHOICES_DEFAULT = 500
EVENT_NUM_DEFAULT = 16


# Random Parameter Ranges
CLUSTER_SZ_RANGE = (1, 5)
CLUSTER_TH_RANGE = (0.1, 0.2)

EVENT_DIR = "./perceptio_data/perceptio_real_fuzz/perceptio_real_event_fuzz"

FUZZING_DIR = "perceptio_real_fuzz"
FUZZING_STUB = "perceptio_real_fuzz"

DEVICE_GROUPS = [
    ["10.0.0.238", "10.0.0.228", "10.0.0.239"],
    ["10.0.0.231", "10.0.0.232", "10.0.0.239"],
    ["10.0.0.233", "10.0.0.236", "10.0.0.239"],
    ["10.0.0.227", "10.0.0.229", "10.0.0.237"],
    ["10.0.0.235", "10.0.0.237", "10.0.0.239"],
    ["10.0.0.234", "10.0.0.239", "10.0.0.237"],
]

DEFAULT_SENSOR_TYPE = "mic"

DEFAULT_TIMESTAMP = "20240813*"

SENSOR_DATA_DIR = "/mnt/nas"


def main(
    key_length=KEY_LENGTH_DEFAULT,
    number_of_choices=NUMBER_OF_CHOICES_DEFAULT,
    data_dir=SENSOR_DATA_DIR,
    sensor_type=DEFAULT_SENSOR_TYPE,
    dev_groups=DEVICE_GROUPS,
    timestamp=DEFAULT_TIMESTAMP,
):
    make_dirs(DATA_DIRECTORY, FUZZING_DIR, FUZZING_STUB)

    fuzzing_dir = f"{DATA_DIRECTORY}/{FUZZING_DIR}/{FUZZING_STUB}"

    groups = load_real_signal_groups(data_dir, dev_groups, sensor_type, timestamp)

    def get_random_parameters():
        cluster_size = random.randint(CLUSTER_SZ_RANGE[0], CLUSTER_SZ_RANGE[1])  # nosec
        cluster_th = random.uniform(CLUSTER_TH_RANGE[0], CLUSTER_TH_RANGE[1])  # nosec
        event_dir, params = load_random_events(EVENT_DIR)
        top_th = params["top_th"]
        bottom_th = params["bottom_th"]
        lump_th = params["lump_th"]
        a = params["a"]

        return (
            top_th,
            bottom_th,
            lump_th,
            a,
            cluster_size,
            cluster_th,
            MICROPHONE_SAMPLING_RATE,
            EVENT_NUM_DEFAULT,
            event_dir,
        )

    def log(params, file_name_stub):
        names = [
            "top_th",
            "bottom_th",
            "lump_th",
            "a",
            "cluster_size",
            "cluster_th",
            "sampling_rate",
            "number_of_events",
            "event_dir",
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

    def func(signals, *params):
        key_size = params[0]
        cluster_sizes_to_check = params[5]
        cluster_th = params[6]
        Fs = params[7]
        number_of_events = params[8]
        return calc_all_event_bits(
            signals,
            process_events,
            number_of_events,
            key_size // 8,
            cluster_sizes_to_check,
            cluster_th,
            Fs,
        )

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        func,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_bit_gen=True,
    )
    evaluator.fuzzing_evaluation(
        groups,
        number_of_choices,
        key_length,
        fuzzing_dir,
        FUZZING_STUB,
        multithreaded=True,
    )


if __name__ == "__main__":
    main()
