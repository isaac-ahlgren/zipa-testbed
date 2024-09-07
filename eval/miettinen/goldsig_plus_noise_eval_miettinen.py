import os
import sys

import numpy as np
from miettinen_tools import (
    MICROPHONE_SAMPLING_RATE,
    adversary_signal,
    get_command_line_args,
    golden_signal,
    miettinen_calc_sample_num,
    miettinen_wrapper_func,
)

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import load_controlled_signal_buffers  # noqa: E402
from evaluator import Evaluator  # noqa: E402

SNAP_SHOT_WIDTH_DEFAULT = 0.1
NO_SNAP_SHOT_WIDTH_DEFAULT = 0.1
ABS_THRESH_DEFAULT = 5e-5
REL_THRESH_DEFAULT = 2
KEY_LENGTH_DEFAULT = 128
TARGET_SNR_DEFAULT = 20
TRIALS_DEFAULT = 100


def main(
    w=SNAP_SHOT_WIDTH_DEFAULT,
    f=NO_SNAP_SHOT_WIDTH_DEFAULT,
    abs_thresh=ABS_THRESH_DEFAULT,
    rel_thresh=REL_THRESH_DEFAULT,
    key_length=KEY_LENGTH_DEFAULT,
    target_snr=TARGET_SNR_DEFAULT,
    trials=TRIALS_DEFAULT,
):
    # Converting time durations to number of samples
    w_in_samples = int(w * MICROPHONE_SAMPLING_RATE)
    f_in_samples = int(f * MICROPHONE_SAMPLING_RATE)

    # Calculating the number of samples needed
    sample_num = miettinen_calc_sample_num(key_length, w_in_samples, f_in_samples)

    # Generating the signals
    signal1 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signal2 = golden_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    adv_signal = adversary_signal(sample_num, MICROPHONE_SAMPLING_RATE)
    signals = load_controlled_signal_buffers(
        [signal1, signal2, adv_signal], target_snr=target_snr, noise=True
    )

    # Defining the bit generation algorithm
    def bit_gen_algo(signal: np.ndarray, *argv) -> np.ndarray:
        """
        Wrapper function to process an array using the Miettinen algorithm.

        :param signal: Input signal array.
        :return: Processed signal array.
        """
        samples = signal.read(argv[4])
        return miettinen_wrapper_func(samples, argv[0], argv[1], argv[2], argv[3])

    # Creating an evaluator object with the bit generation algorithm
    evaluator = Evaluator(bit_gen_algo)
    # Evaluating the signals with the specified number of trials
    evaluator.evaluate_controlled_signals(
        signals, trials, f_in_samples, w_in_samples, rel_thresh, abs_thresh, sample_num
    )
    # Comparing the bit errors for legitimate and adversary signals
    legit_bit_errs, adv_bit_errs = evaluator.cmp_collected_bits(key_length)

    le_avg_be = np.mean(legit_bit_errs)
    adv_avg_be = np.mean(adv_bit_errs)

    # Printing the average bit error rates
    print(f"Legit Average Bit Error Rate: {le_avg_be}")
    print(f"Adversary Average Bit Error Rate: {adv_avg_be}")
    return le_avg_be, adv_avg_be


if __name__ == "__main__":
    args = get_command_line_args(
        snap_shot_width_default=SNAP_SHOT_WIDTH_DEFAULT,
        no_snap_shot_width_default=NO_SNAP_SHOT_WIDTH_DEFAULT,
        absolute_threshold_default=ABS_THRESH_DEFAULT,
        relative_threshold_default=REL_THRESH_DEFAULT,
        key_length_default=KEY_LENGTH_DEFAULT,
        snr_level_default=TARGET_SNR_DEFAULT,
        trials_default=TRIALS_DEFAULT,
    )
    main(*args)
