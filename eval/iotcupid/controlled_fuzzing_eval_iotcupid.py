import os
import random
import sys
from typing import List

import numpy as np
from iotcupid_tools import (
    DATA_DIRECTORY,
    MICROPHONE_SAMPLING_RATE,
    process_events,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    get_fuzzing_command_line_args,
    load_random_events,
    load_controlled_signal_files,
    log_parameters,
    calc_all_event_bits,
    make_dirs,
)
from evaluator import Evaluator  # noqa: E402
from signal_file import Signal_File, Event_File  # noqa: E402

# Static default parameters
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 40
NUMBER_OF_CHOICES_DEFAULT = 500
WRAP_AROUND_LIMIT_DEFAULT = 10
EVENT_NUM_DEFAULT = 16
M_START_DEFAULT = 1
M_END_DEFAULT = 2
M_SEARCHES_DEFAULT = 10


# Random Parameter Ranges
FEATURE_DIM_RANGE = (1, 8)
QUANT_FACTOR_RANGE = (1, 1500)
MEM_THRESH_RANGE = (0.5, 1)
CLUSTER_SZ_RANGE = (1, 5)
CLUSTER_TH_RANGE = (0.1, 0.2)

EVENT_DIR = "./iotcupid_data/iotcupid_controlled_fuzz/iotcupid_controlled_event_fuzz_snr"

FUZZING_DIR = "iotcupid_controlled_fuzz"
FUZZING_STUB = "iotcupid_controlled_fuzz"


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
        cluster_size = random.randint(CLUSTER_SZ_RANGE[0], CLUSTER_SZ_RANGE[1])  # nosec
        cluster_th = random.uniform(CLUSTER_TH_RANGE[0], CLUSTER_TH_RANGE[1])  # nosec
        feature_dim = random.randint(FEATURE_DIM_RANGE[0], FEATURE_DIM_RANGE[1]) # nosec
        quant_factor = random.randint(QUANT_FACTOR_RANGE[0], QUANT_FACTOR_RANGE[1]) # nosec
        mem_thresh = random.randint(MEM_THRESH_RANGE[0], MEM_THRESH_RANGE[1]) # nosec
        event_dir, params = load_random_events(EVENT_DIR + str(target_snr))
        top_th = params["top_th"]
        bottom_th = params["bottom_th"]
        lump_th = params["lump_th"]
        a = params["a"]
        window_sz = params["window_sz"]
        print(f"feature_dim: {feature_dim},mem_thresh: {mem_thresh},
              quant_factor: {quant_factor}, cluster_th: {cluster_th},")

        return (
            top_th,
            bottom_th,
            lump_th,
            a,
            window_sz,
            feature_dim,
            M_START_DEFAULT,
            M_END_DEFAULT,
            M_SEARCHES_DEFAULT,
            mem_thresh,
            quant_factor,
            cluster_size,
            cluster_th,
            MICROPHONE_SAMPLING_RATE,
            EVENT_NUM_DEFAULT,
            event_dir,
        )

    def log(params, file_name_stub):
        names = ["top_th", "bottom_th", "lump_th", "a", 
                 "window_sz", "feature_dim", "m_start", 
                 "m_end", "m_searches", "mem_thresh", 
                 "quant_factor", "cluster_size", "cluster_th", 
                 "sampling_rate", "number_of_events", "event_dir"]
        param_list = [params[0], params[1], params[2], params[3], params[4], 
                      params[5], params[6], params[7], params[8], params[9], 
                      params[10], params[11], params[12], params[13], params[14], 
                      params[15]]
        log_parameters(file_name_stub, names, param_list)

    def func(signals, *params):
        key_size = params[0]
        feature_dim = params[6]
        m_start = params[7]
        m_end = params[8]
        m_searches = params[9]
        mem_thresh = params[10]
        quant_factor = params[11]
        cluster_sizes_to_check = params[12]
        cluster_th = params[13]
        Fs = params[14]
        number_of_events = params[15]
        print(f"key_size: {key_size}, feature_dim: {feature_dim}, m_start: {m_start}, m_end: {m_end}, m_searches: {m_searches}, mem_thresh: {mem_thresh}, quant_factor: {quant_factor}, cluster_sizes_to_check: {cluster_sizes_to_check}, cluster_th: {cluster_th}, Fs: {Fs}, number_of_events: {number_of_events}")
        return calc_all_event_bits(signals, process_events, number_of_events, 
                                   key_size, feature_dim, m_start, m_end, 
                                   m_searches, mem_thresh, quant_factor, 
                                   cluster_sizes_to_check, cluster_th, Fs)

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
    main()