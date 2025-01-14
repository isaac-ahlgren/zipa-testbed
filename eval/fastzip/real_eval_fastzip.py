import os
import random
import sys

from fastzip_tools import DATA_DIRECTORY, calc_all_event_bits_fastzip

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

EVENT_FILE_STUB = "fastzip_event_real"

FUZZING_DIR = "fastzip_real_fuzz"
FUZZING_STUB = "fastzip_real"

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

PARAM_DIR = "../plot_scripts/plot_data/fastzip_real_fuzz"
PARAM_FILE_STUB = "fastzip_real_fuzz"

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
        names = [
            "window_size",
            "overlap_size",
            "n_bits",
            "eqd_delta",
            "ewma",
            "alpha",
            "remove_noise",
            "bias",
            "normalize",
            "power_th",
            "snr_th",
            "peak_th",
            "peak_status",
            "sample_rate"
        ]
        output = [params[name] for name in names]
        output.append(EVENT_NUM_DEFAULT)
        return output

    group_signals, group_params = load_signal_groups(
        groups,
        sensor_type,
        timestamp,
        data_dir,
        param_dir,
        PARAM_FILE_STUB,
        unpack_parameters,
        load_func=wav_file_load,
        event_dir_file_stub=EVENT_FILE_STUB,
        data_dir=DATA_DIRECTORY
    )

    def func(signals, *params):
        key_size = params[0]
        remove_noise = params[7]
        ewma_filter = params[5]
        alpha = params[6]
        bias = params[8]
        n_bits = params[3]
        eqd_delta = params[4]
        print(
            f"key_size: {key_size} ; remove_noise: {remove_noise} ; ewma_filter: {ewma_filter} ; alpha: {alpha} ; bias: {bias} ; n_bits: {n_bits} ; eqd_delta: {eqd_delta}"
        )
        return calc_all_event_bits_fastzip(
            signals, key_size, remove_noise, ewma_filter, alpha, bias, n_bits, eqd_delta
        )


    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(func, event_gen=True, convert_bytes_to_bitstring=False)
    # Evaluating the signals with the specified number of trials
    evaluator.best_parameter_evaluation(
        group_signals, group_params, key_length, DATA_DIRECTORY, FUZZING_STUB
    )


if __name__ == "__main__":
    main()