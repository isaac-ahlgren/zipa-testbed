import argparse
import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    add_gauss_noise,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)
from scipy.io import wavfile


def controlled_sig_plus_noise_eval(
    window_length, band_length, key_length, antialias_freq, target_snr, trials
):
    legit_bit_errs = []
    adv_bit_errs = []

    signal, sr = load_controlled_signal("../controlled_signal.wav")
    adv_signal, sr = load_controlled_signal("../adversary_controlled_signal.wav")
    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, sr, antialias_freq
    )
    index = 0
    adv_index = 0
    for i in range(trials):
        signal_part, index = wrap_around_read(signal, index, sample_num)
        adv_part, adv_index = wrap_around_read(adv_signal, adv_index, sample_num)
        sig1 = add_gauss_noise(signal_part, target_snr)
        sig2 = add_gauss_noise(signal_part, target_snr)
        adv_sig = add_gauss_noise(adv_part, target_snr)
        bits1 = schurmann_wrapper_func(
            sig1, window_length, band_length, sr, antialias_freq
        )
        bits2 = schurmann_wrapper_func(
            sig2, window_length, band_length, sr, antialias_freq
        )
        adv_bits = schurmann_wrapper_func(
            adv_sig, window_length, band_length, sr, antialias_freq
        )
        legit_bit_err = cmp_bits(bits1, bits2, key_length)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = cmp_bits(bits1, adv_bits, key_length)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_err, adv_bit_err


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
            samples_to_read = samples_to_read - samples_can_read
            index = 0
        else:
            buf = buffer[index : index + samples_to_read]
            output = np.append(output, buf)
            index = index + samples_to_read
            samples_to_read = 0
    return output, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wl", "--window_length", type=int, default=16537)
    parser.add_argument("-bl", "--band_length", type=int, default=500)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-snr", "--snr_level", type=float, default=20)
    parser.add_argument("-t", "--trials", type=int, default=1000)

    args = parser.parse_args()
    window_length = getattr(args, "window_length")
    band_length = getattr(args, "band_length")
    key_length = getattr(args, "key_length")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = controlled_sig_plus_noise_eval(
        window_length, band_length, key_length, ANTIALIASING_FILTER, snr_level, trials
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
