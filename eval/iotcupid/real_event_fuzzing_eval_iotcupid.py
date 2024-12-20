import os
import random
import sys

from iotcupid_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    extract_all_events,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    load_real_signal_files,
    log_parameters,
    make_dirs,
    wav_file_load,
)
from evaluator import Evaluator  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
A_LENGTH_RANGE = (0, 1)
TOP_TH_RANGE = (0.00001, 100)
WINDOW_LENGTH_RANGE = (100, MICROPHONE_SAMPLING_RATE * 30)
LUMP_TH_RANGE = (0, 5000)
BOTTOM_TH_MIN_VAL = 0

FUZZING_DIR = "iotcupid_real_fuzz"
FUZZING_STUB = "iotcupid_real_event_fuzz"

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

SENSOR_DATA_DIR = "/home/isaac/test"


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

    signals = load_real_signal_files(
        data_dir, dev_ids, sensor_type, timestamp, load_func=wav_file_load
    )

    def get_random_parameters():
        top_th = random.uniform(TOP_TH_RANGE[0], TOP_TH_RANGE[1])  # nosec
        bottom_th = random.uniform(BOTTOM_TH_MIN_VAL, top_th)  # nosec
        window_sz = random.randint(
            WINDOW_LENGTH_RANGE[0], WINDOW_LENGTH_RANGE[1]
        )  # nosec
        lump_th = random.randint(
            LUMP_TH_RANGE[0], LUMP_TH_RANGE[1] * window_sz
        )  # nosec
        a = random.uniform(A_LENGTH_RANGE[0], A_LENGTH_RANGE[1])  # nosec
        # Calculating the number of samples needed
        return (top_th, bottom_th, lump_th, a, window_sz)

    def log(params, file_name_stub):
        names = ["top_th", "bottom_th", "lump_th", "a", "window_sz"]
        param_list = [params[0], params[1], params[2], params[3], params[4]]
        log_parameters(file_name_stub, names, param_list)

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        extract_all_events,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_gen=True,
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
