import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src
from protocols.fastzip import FastZIP_Protocol  # noqa: E402

SAMPLING_RATE = 50


def fastzip_wrapper_function(
    sensor_arr,
    chunk_size,
    bits,
    power_thr,
    snr_thr,
    peak_thr,
    bias,
    sample_rate,
    eqd_delta,
    ewma_filter=None,
    alpha=None,
    remove_noise=None,
    normalize=None,
):
    return FastZIP_Protocol.fastzip_algo(
        [sensor_arr],
        [chunk_size],
        [bits],
        [power_thr],
        [snr_thr],
        [peak_thr],
        [bias],
        [sample_rate],
        [eqd_delta],
        [ewma_filter],
        [alpha],
        [remove_noise],
        [normalize],
    )


def golden_signal(sample_num, seed):
    np.random.seed(seed)
    output = np.random.rand(sample_num)
    return output


# This is redundant, might as well eliminate
def adversary_signal(sample_num, seed):
    np.random.seed(seed)
    output = np.random.rand(sample_num)
    return output


# Might need to revisit how this calculation is done
def fastzip_calc_sample_num(key_length, window_length):
    return (key_length + 1) * window_length
