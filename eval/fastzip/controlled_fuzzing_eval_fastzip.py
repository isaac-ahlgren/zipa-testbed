import os
import random
import sys
from typing import List

import numpy as np
from fastzip_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    calc_all_event_bits_fastzip,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    get_fuzzing_command_line_args,
    load_random_events,
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

EVENT_DIR = "./fastzip_data/fastzip_controlled_fuzz/fastzip_controlled_event_fuzz_snr"

FUZZING_DIR = "fastzip_controlled_fuzz"
FUZZING_STUB = "fastzip_controlled_fuzz"


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

    signals = [signals]

    def get_random_parameters():
        event_dir, params = load_random_events(EVENT_DIR + str(target_snr))
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
        ewma = False #random.choice([True, False])  # nosec
        alpha = 0.5 #random.uniform(0, 1)  # nosec
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
        print(f"key_size: {key_size} ; remove_noise: {remove_noise} ; ewma_filter: {ewma_filter} ; alpha: {alpha} ; bias: {bias} ; n_bits: {n_bits} ; eqd_delta: {eqd_delta}")
        return calc_all_event_bits_fastzip(signals, key_size, remove_noise, ewma_filter, alpha, bias, n_bits, eqd_delta)

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        func,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_bit_gen=True,
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
    args = get_fuzzing_command_line_args(
        key_length_default=KEY_LENGTH_DEFAULT,
        target_snr_default=TARGET_SNR_DEFAULT,
        number_of_choices_default=NUMBER_OF_CHOICES_DEFAULT,
        wrap_around_limit_default=WRAP_AROUND_LIMIT_DEFAULT,
    )
    main(*args)
