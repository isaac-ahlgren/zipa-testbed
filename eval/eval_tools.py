import glob

import numpy as np

from scipy.io import wavfile


class Signal_Buffer:
    def __init__(self, buf, noise=False, target_snr=20):
        self.signal_buffer = buf
        self.index = 0
        self.noise = noise
        if self.noise:
            self.noise_std = calc_snr_dist_params(buf, target_snr)


    def read(self, samples_to_read):
        samples = samples_to_read

        output = np.array([])
        while samples_to_read != 0:
            samples_can_read = len(self.signal_buffer) - self.index
            if samples_can_read <= samples_to_read:
                buf = self.signal_buffer[self.index : self.index + samples_can_read]
                output = np.append(output, buf)
                samples_to_read = samples_to_read - samples_can_read
                self.index = 0
            else:
                buf = self.signal_buffer[self.index : self.index + samples_to_read]
                output = np.append(output, buf)
                self.index = self.index + samples_to_read
                samples_to_read = 0
        if self.noise:
            noise = np.random.normal(0, self.noise_std, samples)
            output += noise
        return output

    def sync(self, other_signal_buff):
        self.index = other_signal_buff.index


class Signal_File:
    def __init__(
        self, signal_directory, file_names, wrap_around_read=False, load_func=np.loadtxt
    ):
        self.signal_directory = signal_directory
        self.files = glob.glob(file_names, root_dir=signal_directory)
        if len(self.files) == 0:
            print("No files found")
        else:
            self.files.sort()
        self.file_index = 0
        self.start_sample = 0
        self.load_func = load_func
        self.sample_buffer = self.load_func(self.signal_directory + self.files[0])
        self.wrap_around_read = wrap_around_read

    def switch_files(self):
        self.start_sample = 0
        self.file_index += 1
        print("Loading in " + self.signal_directory + self.files[self.file_index])
        if len(self.files) == self.file_index and self.wrap_around_read:
            self.reset()
        else:
            del self.sample_buffer
            self.sample_buffer = self.load_func(
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
        self.sample_buffer = self.load_func(self.signal_directory + self.files[0])

    def sync(self, other_sf):
        self.file_index = other_sf.file_index
        self.start_sample = other_sf.start_sample
        del self.sample_buffer
        self.sample_buffer = self.load_func(
            self.signal_directory + self.files[self.file_index]
        )

def load_controlled_signal(file_name):
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64), sr

def calc_snr_dist_params(signal, target_snr):
    sig_avg_sqr = np.mean(signal) ** 2
    sig_avg_db = 10 * np.log10(sig_avg_sqr)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg_sqr = 10 ** (noise_avg_db / 10)
    return np.sqrt(noise_avg_sqr)

def add_gauss_noise(signal, target_snr):
    noise_std = calc_snr_dist_params(signal, target_snr)
    noise = np.random.normal(0, noise_std, len(signal))
    return signal + noise


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


def events_cmp_bits(fp1, fp2, length_in_bits):
    lowest_bit_error = 100
    for dev1 in fp1:
        for dev2 in fp2:
            bit_err = cmp_bits(dev1, dev2, length_in_bits)
            if bit_err < lowest_bit_error:
                lowest_bit_error = bit_err
    return lowest_bit_error


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
    return -np.log2(max_prob)
