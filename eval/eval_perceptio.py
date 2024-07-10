import sys
import os
sys.path.insert(1, os.getcwd() + "/../src/")

import math
import random
import numpy as np
import pandas as pd
from eval_tools import *
from protocols.perceptio import Perceptio_Protocol

def get_events(arr, top_th, bottom_th, lump_th, a):
    events = Perceptio_Protocol.get_events(arr, a, bottom_th, top_th, lump_th)

    event_features = Perceptio_Protocol.get_event_features(events, signal)

    return event_features

def generate_bits(sf, chunk_size, max_samples, max_events_to_detect, cluster_sizes_to_check, cluster_th, top_th, bottom_th, lump_th, a, Fs, key_size_in_bytes):
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

        fps = Perceptio_Protocol.gen_fingerprints(grouped_events, k, key_size_in_bytes, Fs)
        keys_generated.append(fps)
    return keys_generated, events

def experiment(data_from_devices, ids, iterations, max_samples, Fs):
    byte_length = 16            # Hardcoded to always try to produce keys of a certain length
    max_events_to_detect = 8    # Arbitrarily hardcoded
    cluster_th = 0.1            # Arbitrarily hardcoded
    chunk_size = 48000*3600
 

    file_name = f"perceptio_mic_dev1_{ids[0]}_dev2_{ids[1]}.csv"
    
    min_cluster_size_to_check = 1
    max_cluster_size_to_check = 5
    top_th_max = 10
    top_th_min = 0
    bottom_th_min = 0
    lump_th_max = Fs
    lump_th_min = 10
    a_min = 0
    a_max = 1

    for i in range(iterations):
        cluster_size_to_check = random.randint(min_cluster_size_to_check, max_cluster_size_to_check)
        top_th = random.uniform(top_th_min, top_th_max)
        bottom_th = random.uniform(bottom_th_min, top_th)
        lump_th = random.randint(lump_th_min, lump_th_max)
        a = random.uniform(a_min, a_max)

        device_bits = []
        samples_per_key = None
        for sf in data_from_devices:
            keys_generated, events = generate_bits(sf, chunk_size, max_samples, max_events_to_detect, cluster_size_to_check, cluster_th, top_th, bottom_th, lump_th, a, Fs, byte_length)
            device_bits.append(keys_generated)
            sf.reset()

        avg_bit_err = events_get_average_bit_err(device_bits[0], device_bits[1], bit_length)
        flattened_dev1 = flatten_fingerprints(device_bits[0])
        flattened_dev2 = flatten_fingerprints(device_bits[1])
        min_entropy1_1bit = get_min_entropy(flattened_dev1, bit_length, 1)
        min_entropy2_1bit = get_min_entropy(flattened_dev2, bit_length, 1)
        min_entropy1_4bit = get_min_entropy(flattened_dev1, bit_length, 4)
        min_entropy2_4bit = get_min_entropy(flattened_dev2, bit_length, 4)
        min_entropy1_8bit = get_min_entropy(flattened_dev1, bit_length, 8)
        min_entropy2_8bit = get_min_entropy(flattened_dev2, bit_length, 8)

        results = {"Cluster Size to Check": [cluster_size_to_check],
                   "Top Threshold": [top_th],
                   "Bottom Threshold": [bottom_th],
                   "Lump Threshold": [lump_th],
                   "a": [a],
                   "Average Bit Error": [avg_bit_err],
                   "Min Entropy Device 1 Symbol Size 1 Bit": [min_entropy1_1bit],
                   "Min Entropy Device 2 Symbol Size 1 Bit": [min_entropy2_1bit],
                   "Min Entropy Device 1 Symbol Size 4 Bit": [min_entropy1_4bit],
                   "Min Entropy Device 2 Symbol Size 4 Bit": [min_entropy2_4bit],
                   "Min Entropy Device 1 Symbol Size 8 Bit": [min_entropy1_8bit],
                   "Min Entropy Device 2 Symbol Size 8 Bit": [min_entropy2_8bit]}

        df = pd.DataFrame(results)
        df.to_csv(file_name, mode="a", index=False, header=(not os.path.isfile(file_name)))

def perform_eval():
    iterations = 100
    max_samples = 48000*3600*23
    signal_directory = "/mnt/data/"
    file_name1 = "mic_id_10.0.0.231_date_20240620*.csv"
    file_name2 = "mic_id_10.0.0.232_date_20240620*.csv"
    sf1 = Signal_File(signal_directory, file_name1)
    sf2 = Signal_File(signal_directory, file_name2)

    ids = ["10.0.0.231", "10.0.0.232"]

    data_from_devices = [sf1, sf2]
    experiment(data_from_devices, ids, iterations, max_samples, 48000)

if __name__ == "__main__":
    perform_eval()

