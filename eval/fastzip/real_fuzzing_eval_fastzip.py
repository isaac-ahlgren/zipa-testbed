import os
import random
import sys

import numpy as np
from fastzip_tools import DATA_DIRECTORY, calc_all_event_bits_fastzip

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    get_fuzzing_command_line_args,
    load_random_events,
    load_real_signal_groups,
    log_parameters,
    make_dirs,
    wav_file_load,
)
from evaluator import Evaluator  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 2000
WRAP_AROUND_LIMIT_DEFAULT = 10

# Random Parameter Ranges
WINDOW_SIZE_RANGE = (50, 10000)

MIN_OVERLAP_DEFAULT = 0

MIN_N_BITS_DEFAULT = 10

BIAS_DEFAULT = 0  # no need to worry

MIN_EQD_DELTA_DEFAULT = 1  # dependent of window_size /  n_bits (smaller)

EWMA_FILTER_DEFAULT = None
ALPHA_DEFAULT = None  # between 0 and 1
REMOVE_NOISE_DEFAULT = None
NORMALIZE_DEFAULT = True

POWER_TH_RANGE = (30, 200)

SNR_TH_RANGE = (0.5, 2)

# no need to worry
PEAK_STATUS_DEFAULT = None
NUM_PEAKS_DEFAULT = 0
BIAS_DEFAULT = 0

EVENT_DIR = "./fastzip_data/fastzip_real_fuzz/fastzip_event_real_fuzz"

FUZZING_DIR = "fastzip_real_fuzz"
FUZZING_STUB = "fastzip_real_fuzz"

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

SENSOR_DATA_DIR = "/home/isaac/test"


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

    groups = load_real_signal_groups(
        data_dir, dev_groups, sensor_type, timestamp, load_func=wav_file_load
    )

    def get_random_parameters():
        event_dir, params = load_random_events(EVENT_DIR)
        window_size = params["window_size"]
        overlap_size = params["overlap_size"]
        power_th = params["power_th"]
        snr_th = params["snr_th"]
        peak_th = params["peak_th"]
        sample_rate = params["sample_rate"]
        peak_status = params["peak_status"]
        normalize = params["normalize"]
        alpha = params["alpha"]
        max_bits = key_length if window_size // 2 > key_length else window_size // 2
        n_bits = random.randint(MIN_N_BITS_DEFAULT, max_bits)  # nosec
        eqd_delta = 1
        ewma = random.choice([True, False])  # nosec
        alpha = random.uniform(0, 1)  # nosec
        remove_noise = False #random.choice([True, False])  # nosec
        bias = BIAS_DEFAULT
        return (
            window_size,
            overlap_size,
            n_bits,
            eqd_delta,
            ewma,
            alpha,
            remove_noise,
            bias,
            normalize,
            power_th,
            snr_th,
            peak_th,
            peak_status,
            sample_rate,
            event_dir,
        )

    def log(params, file_name_stub):
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
            "sample_rate",
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
            params[9],
            params[10],
            params[11],
            params[12],
            params[13],
            params[14],
        ]
        for param, name in zip(param_list, names):
            print(f"{name}: {param} ; ", end="")
        print("\n")
        log_parameters(file_name_stub, names, param_list)

    def func(signals, *params) -> np.ndarray:
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

    evaluator = Evaluator(
        func,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_bit_gen=True,
        convert_bytes_to_bitstring=False,
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
