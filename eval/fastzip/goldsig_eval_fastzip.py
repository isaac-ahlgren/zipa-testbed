import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py

import numpy as np
from eval_tools import cmp_bits

from fastzip_tools import (
    SAMPLING_RATE,
    adversary_signal,
    golden_signal,
    fastzip_calc_sample_num,
    fastzip_wrapper_function,
    grab_parameters
)

def parameters(sig):
    sig_copy = np.copy(sig)
    power_threshold, snr_threshold, peaks = grab_parameters(sig_copy)
    return power_threshold, snr_threshold, peaks


#rewrite to work with fastzip
def goldsig_eval(
    w,
    f,
    power_thresh,
    snr_thresh,
    peaks,
    key_length,
    goldsig_sampling_freq,
    trials,
):
    w_in_samples = int(w * goldsig_sampling_freq)
    f_in_samples = int(f * goldsig_sampling_freq)
    legit_bit_errs = []
    adv_bit_errs = []
    sample_num = fastzip_calc_sample_num(
        key_length, w_in_samples, f_in_samples,
    )
    signal = golden_signal(sample_num, goldsig_sampling_freq)
    adv_signal = adversary_signal(sample_num, goldsig_sampling_freq)
    for i in range(trials):
        bits1 = fastzip_wrapper_function(
            signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        bits2 = fastzip_wrapper_function(
            signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        adv_bits = fastzip_wrapper_function(
            adv_signal, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument()

    parser.add_argument("-t", "--trials", type=int, default=1000)