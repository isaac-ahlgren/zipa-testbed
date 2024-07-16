import glob

import numpy as np

# import random
# from datetime import datetime, timedelta


class Signal_File:
    def __init__(self, signal_directory, file_names):
        self.signal_directory = signal_directory
        self.files = glob.glob(file_names, root_dir=signal_directory)
        if len(self.files) == 0:
            print("No files found")
        else:
            self.files.sort()
        self.file_index = 0
        self.start_sample = 0
        print("Loading in " + self.signal_directory + self.files[0])
        self.sample_buffer = np.loadtxt(self.signal_directory + self.files[0])

    def switch_files(self):
        self.start_sample = 0
        self.file_index += 1
        print("Loading in " + self.signal_directory + self.files[self.file_index])
        del self.sample_buffer
        self.sample_buffer = np.loadtxt(
            self.signal_directory + self.files[self.file_index]
        )

    def read(self, samples):
        output = np.array([])

        while samples != 0:
            samples_can_read = len(self.sample_buffer) - self.start_sample
            if samples_can_read <= samples:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples_can_read
                ]
                output = np.append(output, buffer)
                self.switch_files()
                samples -= samples_can_read
            else:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples
                ]
                output = np.append(output, buffer)
                self.start_sample = self.start_sample + samples
                samples = 0
        return output

    def reset(self):
        self.start_sample = 0
        self.file_index = 0
        del self.sample_buffer
        self.sample_buffer = np.loadtxt(self.signal_directory + self.files[0])


def bytes_to_bitstring(b, length):
    import binascii

    bs = bin(int(binascii.hexlify(b), 16))[2:]
    difference = length - len(bs)
    if difference > 0:
        for i in range(difference):
            bs += "0"
    elif difference < 0:
        bs = bs[:length]
    return bs


def cmp_bits(bits1, bits2, length):
    b1 = bytes_to_bitstring(bits1, length)
    b2 = bytes_to_bitstring(bits2, length)
    tot = 0
    for i in range(length):
        if b1[i] != b2[i]:
            tot += 1
    return (tot / length) * 100


def get_bit_err(bits1, bits2, key_length):
    bit_err_over_time = []
    for i in range(len(bits1)):
        bs1 = bytes_to_bitstring(bits1[i], key_length)
        bs2 = bytes_to_bitstring(bits2[i], key_length)
        bit_err = cmp_bits(bs1, bs2)
        bit_err_over_time.append(bit_err)
    return bit_err_over_time


def get_average_bit_err(bits1, bits2, key_length):
    bit_errs = get_bit_err(bits1, bits2, key_length)
    return np.average(bit_errs)


def events_cmp_bits(fp1, fp2, length_in_bits):
    lowest_bit_error = 100
    for dev1 in fp1:
        for dev2 in fp2:
            b1 = bytes_to_bitstring(dev1, length_in_bits)
            b2 = bytes_to_bitstring(dev2, length_in_bits)
            bit_err = cmp_bits(b1, b2)
            if bit_err < lowest_bit_error:
                lowest_bit_error = bit_err
    return lowest_bit_error


def events_get_average_bit_err(fps1, fps2, length_in_bits):
    avg_bit_err = 0
    for i in range(len(fps1)):
        bit_err = events_cmp_bits(bs1, bs2)
        avg_bit_err += bit_err
    return avg_bit_err / len(fps1)


def flatten_fingerprints(fps):
    flattened_fps = []
    for fp in fps:
        for bits in fp:
            flattened_fps.append(bits)
    return flattened_fps


def get_min_entropy(bits, key_length, symbol_size):
    arr = []
    for b in bits:
        bs = bytes_to_bitstring(b, key_length)
        for i in range(0, key_length // symbol_size, symbol_size):
            symbol = bs[i * symbol_size : (i + 1) * symbol_size]
            arr.append(int(symbol, 2))

    hist, bin_edges = np.histogram(arr, bins=2**symbol_size)
    pdf = hist / sum(hist)
    max_prob = np.max(pdf)
    print("Min Entropy:")
    print(max_prob)
    print(hist)
    print()
    return -np.log2(max_prob)
