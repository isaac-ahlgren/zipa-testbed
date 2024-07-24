# import math
import os
import sys

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Miettinen algorithm in /src

from protocols.miettinen import Miettinen_Protocol  # noqa: E402

MICROPHONE_SAMPLING_RATE = 48000


def miettinen_wrapper_func(arr, f, w, rel_thresh, abs_thresh):
    return Miettinen_Protocol.miettinen_algo(arr, f, w, rel_thresh, abs_thresh)


def golden_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * np.pi / fs * i) + 1
    return output


def adversary_signal(sample_num, fs):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output[i] = np.sin(2 * (8000) * np.pi / fs * i)
    return output


def miettinen_calc_sample_num(key_length, w, f):
    return (w + f) * (key_length + 1)
