import os
import sys
from typing import ByteString

import numpy as np
from miettinen_tools import (
    get_command_line_args,
    miettinen_calc_sample_num,
    miettinen_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import (  # noqa: E402
    Signal_Buffer,
    cmp_bits,
    load_controlled_signal,
)
from evaluator import Evaluator  # noqa: E402

if __name__ == "__main__":
    # Define default values
    (w, f, abs_thresh, rel_thresh, key_length, target_snr, trials) = (
        get_command_line_args(
            snap_shot_width_default=5,
            no_snap_shot_width_default=5,
            absolute_threshold_default=5e-15,
            relative_threshold_default=0.1,
            key_length_default=128,
            snr_level_default=20,
            trials_default=100,
        )
    )

# Get command-line arguments


# Loading the controlled signals
legit_signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
adv_signal, sr = load_controlled_signal("../../data/adversary_controlled_signal.wav")

# Converting time durations to number of samples
w_in_samples = int(w * sr)
f_in_samples = int(f * sr)

# Calculating the number of samples needed
sample_num = miettinen_calc_sample_num(key_length, w_in_samples, f_in_samples)

legit_signal_buffer1 = Signal_Buffer(
    legit_signal.copy(), noise=True, target_snr=target_snr
)
legit_signal_buffer2 = Signal_Buffer(
    legit_signal.copy(), noise=True, target_snr=target_snr
)
adv_signal_buffer = Signal_Buffer(adv_signal, noise=True, target_snr=target_snr)

# Grouping the signal buffers into a tuple
signals = (legit_signal_buffer1, legit_signal_buffer2, adv_signal_buffer)


# Defining the bit generation algorithm
def bit_gen_algo(signal: Signal_Buffer) -> ByteString:
    """
    Wrapper function to process an array using the Miettinen algorithm.

    :param signal: Input signal array.
    :return: Processed signal array.
    """
    signal_chunk = signal.read(sample_num)  # Reading a chunk of the signal
    return miettinen_wrapper_func(
        signal_chunk, f_in_samples, w_in_samples, rel_thresh, abs_thresh
    )


# Creating an evaluator object with the bit generation algorithm
evaluator = Evaluator(bit_gen_algo)
# Evaluating the signals with the specified number of trials
evaluator.evaluate(signals, trials)
# Comparing the bit errors for legitimate and adversary signals
legit_bit_errs, adv_bit_errs = evaluator.cmp_func(cmp_bits, key_length)

# Printing the average bit error rates
print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
