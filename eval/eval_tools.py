import glob
from typing import Callable, List, Tuple

import numpy as np
from scipy.io import wavfile


class Signal_Buffer:
    def __init__(self, buf: np.ndarray, noise: bool = False, target_snr: int = 20):
        """
        Initialize a buffer to manage signal data with optional noise addition.

        :param buf: Array containing the signal data.
        :param noise: If true, Gaussian noise is added to the output based on the target SNR.
        :param target_snr: The signal-to-noise ratio used to calculate noise level.
        """
        self.signal_buffer = buf
        self.index = 0
        self.noise = noise
        if self.noise:
            self.noise_std = calc_snr_dist_params(buf, target_snr)

    def read(self, samples_to_read: int) -> np.ndarray:
        """
        Read a specific number of samples from the buffer, adding noise if specified.

        :param samples_to_read: The number of samples to read from the buffer.
        :return: An array of the read samples, possibly with noise added.
        """
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

    def sync(self, other_signal_buff: "Signal_Buffer"):
        """
        Synchronize this buffer's index with another signal buffer's index.

        :param other_signal_buff: Another Signal_Buffer instance to synchronize with.
        """
        self.index = other_signal_buff.index


class Signal_File:
    def __init__(
        self,
        signal_directory: str,
        file_names: str,
        wrap_around_read: bool = False,
        load_func: Callable = np.loadtxt,
    ):
        """
        Initialize a file-based signal manager that can handle multiple signal files.

        :param signal_directory: The directory containing signal files.
        :param file_pattern: The glob pattern to match files within the directory.
        :param wrap_around_read: If True, wraps around to the first file after the last file is read.
        :param load_func: The function used to load signal data from a file.
        """
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
        """
        Switch to the next file in the directory or wrap around if enabled.
        """
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

    def read(self, samples: int) -> np.ndarray:
        """
        Read a specified number of samples across multiple files.

        :param samples: Number of samples to read.
        :return: Array containing the read samples.
        """
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
        """
        Reset the reader to the start of the first file.
        """
        self.start_sample = 0
        self.file_index = 0
        del self.sample_buffer
        self.sample_buffer = self.load_func(self.signal_directory + self.files[0])

    def sync(self, other_sf: "Signal_File"):
        """
        Synchronize this file reader with another, matching the current file and sample index.

        :param other_sf: Another Signal_File instance to synchronize with.
        """
        self.file_index = other_sf.file_index
        self.start_sample = other_sf.start_sample
        del self.sample_buffer
        self.sample_buffer = self.load_func(
            self.signal_directory + self.files[self.file_index]
        )


def load_controlled_signal(file_name: str) -> Tuple[np.ndarray, int]:
    """
    Load a controlled signal from a WAV file.

    :param file_name: The path to the WAV file.
    :return: A tuple containing the signal data as a numpy array and the sample rate.
    """
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64), sr


def calc_snr_dist_params(signal: np.ndarray, target_snr: float) -> float:
    """
    Calculate the noise standard deviation for a given signal and target SNR.

    :param signal: The input signal array.
    :param target_snr: The desired signal-to-noise ratio in dB.
    :return: The calculated noise standard deviation.
    """
    sig_avg_sqr = np.mean(signal) ** 2
    sig_avg_db = 10 * np.log10(sig_avg_sqr)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg_sqr = 10 ** (noise_avg_db / 10)
    return np.sqrt(noise_avg_sqr)


def add_gauss_noise(signal: np.ndarray, target_snr: float) -> np.ndarray:
    """
    Add Gaussian noise to a signal based on a target SNR.

    :param signal: The input signal array.
    :param target_snr: The desired signal-to-noise ratio in dB.
    :return: The signal with added Gaussian noise.
    """
    noise_std = calc_snr_dist_params(signal, target_snr)
    noise = np.random.normal(0, noise_std, len(signal))
    return signal + noise


def bytes_to_bitstring(b: bytes, length: int) -> str:
    """
    Convert bytes to a bitstring of a specified length.

    :param b: The bytes object to convert.
    :param length: The desired length of the bitstring.
    :return: A bitstring representation of the bytes.
    """
    import binascii

    bs = bin(int(binascii.hexlify(b), 16))[2:]
    difference = length - len(bs)
    if difference > 0:
        for i in range(difference):
            bs += "0"
    elif difference < 0:
        bs = bs[:length]
    return bs


def cmp_bits(bits1: bytes, bits2: bytes, length: int) -> float:
    """
    Compare two bitstrings and calculate the percentage of differing bits.

    :param bits1: The first bitstring.
    :param bits2: The second bitstring.
    :param length: The length of the bitstrings to compare.
    :return: The percentage of differing bits.
    """
    b1 = bytes_to_bitstring(bits1, length)
    b2 = bytes_to_bitstring(bits2, length)
    tot = 0
    for i in range(length):
        if b1[i] != b2[i]:
            tot += 1
    return (tot / length) * 100


def get_bit_err(bits1: List[bytes], bits2: List[bytes], key_length: int) -> List[float]:
    """
    Calculate bit errors over time between two lists of bitstrings.

    :param bits1: List of bitstrings from the first device.
    :param bits2: List of bitstrings from the second device.
    :param key_length: The length of each bitstring.
    :return: A list of bit error percentages for each corresponding pair of bitstrings.
    """
    bit_err_over_time = []
    for i in range(len(bits1)):
        bs1 = bytes_to_bitstring(bits1[i], key_length)
        bs2 = bytes_to_bitstring(bits2[i], key_length)
        bit_err = cmp_bits(bs1, bs2)
        bit_err_over_time.append(bit_err)
    return bit_err_over_time


def events_cmp_bits(fp1: List[bytes], fp2: List[bytes], length_in_bits: int) -> float:
    """
    Compare two lists of bitstrings and find the lowest bit error.

    :param fp1: List of bitstrings from the first device.
    :param fp2: List of bitstrings from the second device.
    :param length_in_bits: The length to consider for each bitstring comparison.
    :return: The lowest bit error percentage found between all pairs.
    """
    lowest_bit_error = 100
    for dev1 in fp1:
        for dev2 in fp2:
            bit_err = cmp_bits(dev1, dev2, length_in_bits)
            if bit_err < lowest_bit_error:
                lowest_bit_error = bit_err
    return lowest_bit_error


def flatten_fingerprints(fps: List[List[bytes]]) -> List[bytes]:
    """
    Flatten a list of lists of fingerprints into a single list of fingerprints.

    :param fps: A list of lists, where each inner list contains fingerprints.
    :return: A single flattened list containing all fingerprints.
    """
    flattened_fps = []
    for fp in fps:
        for bits in fp:
            flattened_fps.append(bits)
    return flattened_fps


def get_min_entropy(bits: List[bytes], key_length: int, symbol_size: int) -> float:
    """
    Calculate the minimum entropy of a list of bitstrings based on symbol size.

    :param bits: List of bitstrings.
    :param key_length: The total length of each bitstring.
    :param symbol_size: The size of each symbol in bits.
    :return: The minimum entropy observed across all symbols.
    """
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
