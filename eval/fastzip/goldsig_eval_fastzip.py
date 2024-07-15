import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py

import numpy as np
from eval_tools import cmp_bits

from fastzip_tools import (
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    golden_signal,
    fastzip_calc_sample_num,
    fastzip_wrapper_func,
)

#rewrite to work with fastzip
def goldsig_eval(
    w,
    f,
    rel_thresh,
    abs_thresh,
    key_length,
    goldsig_sampling_freq,
    trials,
):
    w_in_samples = int(w * goldsig_sampling_freq)
    f_in_samples = int(f * goldsig_sampling_freq)
    legit_bit_errs = []
    adv_bit_errs = []
    sample_num = miettinen_calc_sample_num(
        key_length, w_in_samples, f_in_samples,
    )
    signal = golden_signal(sample_num, goldsig_sampling_freq)
    adv_signal = adversary_signal(sample_num, goldsig_sampling_freq)
    for i in range(trials):
        bits1 = miettinen_wrapper_func(
            signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        bits2 = miettinen_wrapper_func(
            signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        adv_bits = miettinen_wrapper_func(
            adv_signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs

