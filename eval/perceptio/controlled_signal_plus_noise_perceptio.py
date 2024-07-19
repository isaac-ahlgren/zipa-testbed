import argparse
import os
import sys

import numpy as np
from perceptio_tools import (
    adversary_signal,
    gen_min_events,
    generate_bits,
    golden_signal,
)
from scipy.io import wavfile

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import events_cmp_bits  # noqa: E402


# WIP
def goldsig_eval(
    top_th,
    bottom_th,
    lump_th,
    a,
    cluster_sizes_to_check,
    cluster_th,
    min_events,
    Fs,
    key_size_in_bytes,
    chunk_size,
    trials,
):
    legit_bit_errs = []
    adv_bit_errs = []

    signal, sr = load_controlled_signal("../../data/controlled_signal.wav")
    adv_signal, sr = load_controlled_signal(
        "../../data/adversary_controlled_signal.wav"
    )

    for i in range(trials):
        signal_events, signal_event_features = gen_min_events(
            golden_signal, chunk_size, min_events, top_th, bottom_th, lump_th, a
        )
        adv_events, adv_event_features = gen_min_events(
            adversary_signal, chunk_size, min_events, top_th, bottom_th, lump_th, a
        )
        bits1, grouped_events1 = generate_bits(
            signal_events,
            signal_event_features,
            cluster_sizes_to_check,
            cluster_th,
            Fs,
            key_size_in_bytes,
        )
        bits2, grouped_events2 = generate_bits(
            signal_events,
            signal_event_features,
            cluster_sizes_to_check,
            cluster_th,
            Fs,
            key_size_in_bytes,
        )
        adv_bits, grouped_events_adv = generate_bits(
            adv_events,
            adv_event_features,
            cluster_sizes_to_check,
            cluster_th,
            Fs,
            key_size_in_bytes,
        )
        legit_bit_err = events_cmp_bits(bits1, bits2, key_size_in_bytes * 8)
        legit_bit_errs.append(legit_bit_err)
        adv_bit_err = events_cmp_bits(bits1, adv_bits, key_size_in_bytes * 8)
        adv_bit_errs.append(adv_bit_err)
    return legit_bit_errs, adv_bit_errs


def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64) + 2**16, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tt", "--top_threshold", type=float, default=6)
    parser.add_argument("-bt", "--bottom_threshold", type=float, default=4)
    parser.add_argument("-lt", "--lump_threshold", type=int, default=4)
    parser.add_argument("-a", "--ewma_a", type=float, default=0.75)
    parser.add_argument("-cl", "--cluster_sizes_to_check", type=int, default=4)
    parser.add_argument("-min", "--minimum_events", type=int, default=16)
    parser.add_argument("-fs", "--sampling_frequency", type=float, default=10000)
    parser.add_argument("-ch", "--chunk_size", type=int, default=100)
    parser.add_argument("-kl", "--key_length", type=int, default=128)
    parser.add_argument("-t", "--trials", type=int, default=100)

    args = parser.parse_args()
    top_th = getattr(args, "top_threshold")
    bottom_th = getattr(args, "bottom_threshold")
    lump_th = getattr(args, "lump_threshold")
    a = getattr(args, "ewma_a")
    cluster_sizes_to_check = getattr(args, "cluster_sizes_to_check")
    cluster_th = 0.1
    min_events = getattr(args, "minimum_events")
    Fs = getattr(args, "sampling_frequency")
    chunk_size = getattr(args, "chunk_size")
    key_size_in_bytes = getattr(args, "key_length") // 8
    trials = getattr(args, "trials")

    legit_bit_errs, adv_bit_errs = goldsig_eval(
        top_th,
        bottom_th,
        lump_th,
        a,
        cluster_sizes_to_check,
        cluster_th,
        min_events,
        Fs,
        key_size_in_bytes,
        chunk_size,
        trials,
    )
    print(f"Legit Average Bit Error Rate: {np.mean(legit_bit_errs)}")
    print(f"Adversary Average Bit Error Rate: {np.mean(adv_bit_errs)}")
