import argparse
import os
import sys

# Gives us path to eval_tools.py
sys.path.insert(1, os.getcwd() + "/..")

import numpy as np 
from schurmann_tools import (
    ANTIALIASING_FILTER,
    schurmann_calc_sample_num,
    schurmann_wrapper_func,
)
from scipy.io import wavfile

sys.path.insert(1, os.getcwd() + "/..")
from eval_tools import Signal_Buffer, add_gauss_noise, cmp_bits  # noqa: E402
from evaluator import Evaluator # noqa: E402

def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64), sr


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
    target_snr = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )
    legit_signal_buffer1 = Signal_Buffer(legit_signal.copy())
    legit_signal_buffer2 = Signal_Buffer(legit_signal.copy())
    adv_signal_buffer = Signal_Buffer(adv_signal)

    signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)

    sample_num = schurmann_calc_sample_num(
        key_length, window_length, band_length, sr, ANTIALIASING_FILTER)

    def bit_gen_algo(signal):
        signal_chunk = signal.read(sample_num)
        noisy_signal = add_gauss_noise(signal_chunk, target_snr)
        return schurmann_wrapper_func(noisy_signal, window_length, band_length, sr, ANTIALIASING_FILTER)

    evaluator = Evaluator(bit_gen_algo)
    evaluator.evaluate(signals, trials)
    legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)
    
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
