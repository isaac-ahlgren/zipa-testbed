import sys
import os
sys.path.insert(1, os.getcwd() +"/..") # Gives us path to eval_tools.py
import numpy as np

from schurmann_tools import schurmann_wrapper_func, schurmann_calc_sample_num, golden_signal, MICROPHONE_SAMPLING_RATE, ANTIALIASING_FILTER
from eval_tools import cmp_bits

def goldsig_eval(window_length, band_length, key_length, goldsig_sampling_freq, antialias_freq, trials):
    bit_errs = []
    sample_num = schurmann_calc_sample_num(key_length, window_length, band_length, goldsig_sampling_freq, antialias_freq)
    signal = golden_signal(sample_num, goldsig_sampling_freq)
    for i in range(trials):
        bits1 = schurmann_wrapper_func(signal, window_length, band_length, goldsig_sampling_freq, antialias_freq)
        bits2 = schurmann_wrapper_func(signal, window_length, band_length, goldsig_sampling_freq, antialias_freq)
        bit_err = cmp_bits(bits1, bits2, key_length)
        bit_errs.append(bit_err)
    return bit_errs

if __name__ == "__main__":
    window_length = 16537
    band_length = 500
    key_length = 128
    trials = 1000

    bit_errs = goldsig_eval(window_length, band_length, key_length, MICROPHONE_SAMPLING_RATE, ANTIALIASING_FILTER, trials)
    print(f"Average Bit Error Rate: {np.mean(bit_errs)}")
