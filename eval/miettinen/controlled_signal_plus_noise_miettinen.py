import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from miettinen_tools import (
    add_gauss_noise,
    miettinen_calc_sample_num,
    miettinen_wrapper_func,
)
from scipy.io import wavfile


def controlled_signal_plus_noise_eval(
    w,
    f,
    rel_thresh,
    abs_thresh,
    key_length,
    target_snr,
    trials,
):
    legit_bit_errs = []
    adv_bit_errs = []

    signal, sr = load_controlled_signal("../controlled_signal.wav")
    adv_signal, sr = load_controlled_signal("../adversary_controlled_signal.wav")
    w_in_samples = int(w * sr)
    f_in_samples = int(f * sr)
    sample_num = miettinen_calc_sample_num(
        key_length,
        w_in_samples,
        f_in_samples,
    )
    index = 0
    adv_index = 0
    for i in range(trials):
        signal_part, index = wrap_around_read(signal, index, sample_num)
        adv_part, adv_index = wrap_around_read(adv_signal, adv_index, sample_num)
        sig1 = add_gauss_noise(signal_part, target_snr)
        sig2 = add_gauss_noise(signal_part, target_snr)
        adv_sig = add_gauss_noise(adv_part, target_snr)
        bits1 = miettinen_wrapper_func(
            sig1, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        bits2 = miettinen_wrapper_func(
            sig2, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        adv_bits = miettinen_wrapper_func(
            adv_sig, f_in_samples, w_in_samples, rel_thresh, abs_thresh
        )
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs


def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64) + 2**16, sr


def wrap_around_read(buffer, index, samples_to_read):
    output = np.array([])
    while samples_to_read != 0:
        samples_can_read = len(buffer) - index
        if samples_can_read <= samples_to_read:
            buf = buffer[index : index + samples_can_read]
            output = np.append(output, buf)
            samples_to_read -= samples_can_read
            index = 0
        else:
            buf = buffer[index : index + samples_to_read]
            output = np.append(output, buf)
            index = index + samples_to_read
            samples_to_read = 0
    return output, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--snap_shot_width", type=float, default=5)
    parser.add_argument("-f", "--no_snap_shot_width", type=float, default=5)
    parser.add_argument("-at", "--absolute_threshold", type=float, default=0.5)
    parser.add_argument("-rt", "--relative_threshold", type=float, default=0.1)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-snr", "--snr_level", type=float, default=20)
    parser.add_argument("-t", "--trials", type=int, default=100)

    args = parser.parse_args()
    w = getattr(args, "snap_shot_width")
    f = getattr(args, "no_snap_shot_width")
    abs_thresh = getattr(args, "absolute_threshold")
    rel_thresh = getattr(args, "relative_threshold")
    key_length = getattr(args, "key_length")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = controlled_signal_plus_noise_eval(
        w,
        f,
        rel_thresh,
        abs_thresh,
        key_length,
        snr_level,
        trials,
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
