# import math
import os
import sys
import random

sys.path.insert(
    1, os.getcwd() + "/../../src/"
) 

import numpy as np

from protocols.perceptio import Perceptio_Protocol

def get_events(arr, top_th, bottom_th, lump_th, a):
    events = Perceptio_Protocol.get_events(arr, a, bottom_th, top_th, lump_th)

    event_features = Perceptio_Protocol.get_event_features(events, signal)

    return event_features

def generate_bits(
    sf,
    chunk_size,
    max_samples,
    max_events_to_detect,
    cluster_sizes_to_check,
    cluster_th,
    top_th,
    bottom_th,
    lump_th,
    a,
    Fs,
    key_size_in_bytes,
):
    events = []
    iterations = max_samples // chunk_size
    for i in range(iterations):
        samples = sf.read(max_samples)
        e = get_events(samples, top_th, bottom_th, lump_th, a)
        del samples
        events.extend(e)

    number_of_events = len(events)
    number_of_keys = number_of_events // max_events_to_detect

    for i in range(number_of_keys):
        labels, k = Perceptio_Protocol.kmeans_w_elbow_method(
            event_features, cluster_sizes_to_check, cluster_th
        )

        grouped_events = Perceptio_Protocol.group_events(events, labels, k)

        fps = Perceptio_Protocol.gen_fingerprints(
            grouped_events, k, key_size_in_bytes, Fs
        )
        keys_generated.append(fps)
    return keys_generated, events

golden_rng = np.random.default_rng(0)
def golden_signal(sample_num):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output.append(golden_rng.integers(0,4))
    return output


adv_rng = np.random.default_rng(12345)
def adversary_signal(sample_num):
    output = np.zeros(sample_num)
    for i in range(sample_num):
        output.append(adv_rng.integers(0,4))
    return output
    

def add_gauss_noise(signal, target_snr):
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg), len(signal))
    return signal + noise
