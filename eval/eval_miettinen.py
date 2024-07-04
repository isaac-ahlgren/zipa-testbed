import sys
import os
sys.path.insert(1, os.getcwd() + "/../src/")

import math
import random
import numpy as np
import pandas as pd
from eval_tools import *
from protocols.miettinen import Miettinen_Protocol

def miettinen_wrapper_func(arr, f, w, rel_thresh, abs_thresh):
    return Miettinen_Protocol.miettinen_algo(arr, f, w, rel_thresh, abs_thresh)

def generate_bits(sf, bit_length, f, w, rel_thresh, abs_thresh, max_samples):
    samples_per_key = (w + f) * (bit_length + 1)

    number_of_keys = max_samples // samples_per_key
    keys_generated = []
    for i in range(number_of_keys):
        samples = sf.read(samples_per_key)
        key = miettinen_wrapper_func(samples, f, w, rel_thresh, abs_thresh)
        print(len(key))
        keys_generated.append(key)
    return keys_generated, samples_per_key

def experiment(data_from_devices, ids, iterations, max_samples):
    bit_length = 128            # Hardcoded to always try to produce keys of a certain length

    file_name = f"miettinen_dev1_{ids[0]}_dev2_{ids[1]}.csv"

    min_f = 5000
    max_f = 48000*10
    min_w = 5000
    max_w = 48000*10
    min_rel_thresh = -10000
    max_rel_thresh = 10000
    min_abs_thresh = -30
    max_abs_thresh = 30    

    for i in range(iterations):
        f = random.randint(min_f, max_f)
        w = random.randint(min_w, max_w)
        rel_thresh = random.uniform(min_rel_thresh, max_rel_thresh)
        abs_thresh = random.uniform(min_abs_thresh, max_abs_thresh)

        device_bits = []
        samples_per_key = None
        for sf in data_from_devices:
            bits, samples_per_key = generate_bits(sf, bit_length, f, w, rel_thresh, abs_thresh, max_samples)
            device_bits.append(bits)
            sf.reset()
        avg_bit_err = get_average_bit_err(device_bits[0], device_bits[1], bit_length)
        min_entropy1_1bit = get_min_entropy(device_bits[0], bit_length, 1)
        min_entropy2_1bit = get_min_entropy(device_bits[1], bit_length, 1)
        min_entropy1_4bit = get_min_entropy(device_bits[0], bit_length, 4)
        min_entropy2_4bit = get_min_entropy(device_bits[1], bit_length, 4)
        min_entropy1_8bit = get_min_entropy(device_bits[0], bit_length, 8)
        min_entropy2_8bit = get_min_entropy(device_bits[1], bit_length, 8)

        results = {"f": [f],
                   "w": [w],
                   "rel_thresh": [rel_thresh],
                   "abs_thresh": [abs_thresh],
                   "Samples Per Key": [samples_per_key],
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
    experiment(data_from_devices, ids, iterations, max_samples)

if __name__ == "__main__":
    perform_eval()

