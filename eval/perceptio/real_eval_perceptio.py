import os
import random
import sys

from perceptio_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    process_events,
    unpack_event_parameters,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    load_signal_groups,
    wav_file_load,
)
from evaluator import Evaluator  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 256
NUMBER_OF_CHOICES_DEFAULT = 2000
WRAP_AROUND_LIMIT_DEFAULT = 10

EVENT_NUM_DEFAULT = 5

EVENT_DIR = ""
EVENT_FILE_STUB = "perceptio_event_real_longitudinal"

FUZZING_DIR = "perceptio_real_fuzz"
FUZZING_STUB = "perceptio_real"

DATA_STUB = "perceptio_real_longitudinal"

DEFAULT_GROUPS = [
    ["10.0.0.238", "10.0.0.228", "10.0.0.239"],
    ["10.0.0.231", "10.0.0.232", "10.0.0.239"],
    ["10.0.0.233", "10.0.0.236", "10.0.0.239"],
    ["10.0.0.227", "10.0.0.229", "10.0.0.237"],
    ["10.0.0.235", "10.0.0.237", "10.0.0.239"],
    ["10.0.0.234", "10.0.0.239", "10.0.0.237"],
]

DEFAULT_SENSOR_TYPE = "mic"

DEFAULT_TIMESTAMP = "202408*.wav"

SENSOR_DATA_DIR = "/mnt/nas"

PARAM_DIR = "../plot_scripts/plot_data/perceptio_real_fuzz"
PARAM_FILE_STUB = "perceptio_real_fuzz"

def main(
    key_length=KEY_LENGTH_DEFAULT,
    data_dir=SENSOR_DATA_DIR,
    param_dir=PARAM_DIR,
    sensor_type=DEFAULT_SENSOR_TYPE,
    groups=DEFAULT_GROUPS,
    timestamp=DEFAULT_TIMESTAMP,
):
    if not os.path.isdir(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)

    def unpack_parameters(params):
        names = ["top_th",
                "bottom_th",
                "lump_th",
                "a",
                "cluster_size",
                "cluster_th"
                ]
        output = [params[name] for name in names]
        output.extend([MICROPHONE_SAMPLING_RATE, EVENT_NUM_DEFAULT])
        return output

    group_signals, group_params = load_signal_groups(
        groups,
        sensor_type,
        timestamp,
        data_dir,
        param_dir,
        PARAM_FILE_STUB,
        unpack_event_parameters,
        load_func=wav_file_load,
        event_dir_file_stub=EVENT_FILE_STUB
    )

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
    evaluator = Evaluator(func, event_gen=True)
    # Evaluating the signals with the specified number of trials
    evaluator.best_parameter_evaluation(
        group_signals, group_params, key_length, DATA_DIRECTORY, FUZZING_STUB
    )


if __name__ == "__main__":
    main()