import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src
from protocols.fastzip import FastZIP_Protocol  # noqa: E402

SAMPLING_RATE = 50


def manage_overlapping_chunks(signal_buffer, chunk_size, overlap_size):
    previous_chunk = np.array([])

    while True:
        if len(previous_chunk) < overlap_size:
            new_data = signal_buffer.read(chunk_size)
            if new_data is None:
                break
        else:
            new_data = signal_buffer.read(chunk_size - overlap_size)

        if new_data is None:
            break

        if len(previous_chunk) >= overlap_size:
            current_chunk = np.concatenate((previous_chunk[-overlap_size:], new_data))
        else:
            current_chunk = new_data

        yield current_chunk
        previous_chunk = current_chunk


def fastzip_wrapper_function(
    sensor_arr,
    bits,
    power_thr,
    snr_thr,
    peak_thr,
    bias,
    sample_rate,
    eqd_delta,
    peak_status=None,
    ewma_filter=None,
    alpha=None,
    remove_noise=None,
    normalize=None,
):
    return FastZIP_Protocol.fastzip_algo(
        [sensor_arr],
        [bits],
        [power_thr],
        [snr_thr],
        [peak_thr],
        [bias],
        [sample_rate],
        [eqd_delta],
        [peak_status],
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
