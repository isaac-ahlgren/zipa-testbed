import os
import random
import sys
from typing import List

import numpy as np
from fastzip_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    fastzip_wrapper_function,
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

    def get_random_parameters():
        window_size = random.randint(WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1]) # nosec
        overlap_size = random.randint(MIN_OVERLAP_DEFAULT, window_size // 2)  # nosec
        max_bits = key_length if window_size // 2 > key_length else window_size // 2
        n_bits = random.randint(MIN_N_BITS_DEFAULT, max_bits)  # nosec
        max_eqd_delta = np.ceil(window_size / n_bits)
        eqd_delta = random.randint(MIN_EQD_DELTA_DEFAULT, max_eqd_delta)  # nosec
        ewma = random.choice([True, False])  # nosec
        alpha = random.uniform(0, 1)  # nosec
        remove_noise = random.choice([True, False])  # nosec
        normalize = random.choice([True, False])  # nosec
        power_th = random.uniform(POWER_TH_RANGE[0], POWER_TH_RANGE[1])  # nosec
        snr_th = random.uniform(SNR_TH_RANGE[0], SNR_TH_RANGE[1])  # nosec
        return (
            window_size,
            overlap_size,
            n_bits,
            eqd_delta,
            ewma,
            alpha,
            remove_noise,
            normalize,
            power_th,
            snr_th,
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
            "normalize",
            "power_th",
            "snr_th",
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
        ]
        log_parameters(file_name_stub, names, param_list)

    def bit_gen_algo(signal: Signal_File, *args: List) -> np.ndarray:
        """
        Generates bits based on the analysis of overlapping chunks from a signal.

        :param signal: The signal buffer to process.
        :type signal: Signal_Buffer
        :return: A byte string of the generated bits up to the specified key length.
        :rtype: ByteString
        """
        output, samples_read = fastzip_wrapper_function(
            signal,
            args[2],
            args[0],
            args[1],
            args[8],
            args[9],
            NUM_PEAKS_DEFAULT,
            BIAS_DEFAULT,
            args[3],
            MICROPHONE_SAMPLING_RATE,
            KEY_LENGTH_DEFAULT,
            PEAK_STATUS_DEFAULT,
            args[4],
            args[5],
            args[6],
            args[7],
        )
        if len(output) != KEY_LENGTH_DEFAULT // 8:
            output = None
        return output, samples_read

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(
        bit_gen_algo,
        random_parameter_func=get_random_parameters,
        parameter_log_func=log,
        event_driven=False,
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
