import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from miettinen_tools import (
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    golden_signal,
    miettinen_calc_sample_num,
    miettinen_wrapper_func,
)


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
        key_length,
        w_in_samples,
        f_in_samples,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--snap_shot_width", type=float, default=5)
    parser.add_argument("-f", "--no_snap_shot_width", type=float, default=5)
    parser.add_argument("-at", "--absolute_threshold", type=float, default=0.5)
    parser.add_argument("-rt", "--relative_threshold", type=float, default=0.1)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    args = parser.parse_args()
    w = getattr(args, "snap_shot_width")
    f = getattr(args, "no_snap_shot_width")
    abs_thresh = getattr(args, "absolute_threshold")
    rel_thresh = getattr(args, "relative_threshold")
    key_length = getattr(args, "key_length")
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = goldsig_eval(
        w,
        f,
        rel_thresh,
        abs_thresh,
        key_length,
        MICROPHONE_SAMPLING_RATE,
        trials,
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
