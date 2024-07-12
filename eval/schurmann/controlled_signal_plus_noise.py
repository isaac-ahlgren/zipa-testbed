import os
import sys

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
import numpy as np
from eval_tools import cmp_bits
from schurmann_tools import (
    ANTIALIASING_FILTER,
    MICROPHONE_SAMPLING_RATE,
    add_gauss_noise,
    golden_signal,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)
from scipy.io import wavfile


def controlled_sig_plus_noise_eval(
    window_length, band_length, key_length, antialias_freq, target_snr, trials
):
    bit_errs = []
    signal, sr = load_controlled_signal("../controlled_signal.wav")
    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, sr, antialias_freq
    )
    keys = len(signal) // sample_num
    for i in range(trials):
        for j in range(keys):
            signal_part = signal[j * sample_num : (j + 1) * sample_num]
            sig1 = add_gauss_noise(signal_part, target_snr)
            sig2 = add_gauss_noise(signal_part, target_snr)
            bits1 = schurmann_wrapper_func(
                sig1, window_length, band_length, sr, antialias_freq
            )
            bits2 = schurmann_wrapper_func(
                sig2, window_length, band_length, sr, antialias_freq
            )
            bit_err = cmp_bits(bits1, bits2, key_length)
            bit_errs.append(bit_err)
    return bit_errs


def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64) + 2**16, sr


if __name__ == "__main__":
    window_length = 16537
    band_length = 500
    key_length = 128
    trials = 1000
    target_snr = 40

    bit_errs = controlled_sig_plus_noise_eval(
        window_length, band_length, key_length, ANTIALIASING_FILTER, target_snr, trials
    )
    print(f"Average Bit Error Rate: {np.mean(bit_errs)}")
