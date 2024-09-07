import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from signal_file import (
    Noisy_File,
    Signal_Buffer,
    Signal_File,
    Wrap_Around_File,
)


def calc_snr_dist_params(signal: np.ndarray, target_snr: float) -> float:
    """
    Calculate the noise standard deviation for a given signal and target SNR.

    :param signal: The input signal array.
    :param target_snr: The desired signal-to-noise ratio in dB.
    :return: The calculated noise standard deviation.
    """
    sig_sqr_sum = np.mean(signal**2)
    sig_db = 10 * np.log10(sig_sqr_sum)
    noise_db = sig_db - target_snr
    noise_avg_sqr = 10 ** (noise_db / 10)
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


def gen_id():
    return random.randint(0, 2**64)  # nosec


def calc_all_bits(signal: Signal_File, bit_gen_algo_wrapper, *argv):
    bits = []
    while not signal.get_finished_reading():
        b = bit_gen_algo_wrapper(signal, *argv)
        if b is not None:
            bits.append(b)
    return bits


def calc_all_events(signal: Signal_File, event_gen_algo_wrapper):
    pass


def load_controlled_signal(file_name: str) -> Tuple[np.ndarray, int]:
    """
    Load a controlled signal from a WAV file.

    :param file_name: The path to the WAV file.
    :return: A tuple containing the signal data as a numpy array and the sample rate.
    """
    sr, data = wavfile.read(file_name)
    return data.astype(np.int64)


def wrap_signal_file(
    sf, noise=False, target_snr=None, wrap_around=False, wrap_around_limit=None
):
    if wrap_around:
        sf = Wrap_Around_File(sf, wrap_around_limit=wrap_around_limit)
    if noise:
        sf = Noisy_File(sf, target_snr)
    return sf


def load_signal_files(
    dir,
    files,
    ids,
    load_func=np.loadtxt,
    noise=False,
    target_snr=None,
    wrap_around=False,
    wrap_around_limit=None,
):
    sfs = []
    for file, id in zip(files, ids):
        sf = Signal_File(dir, file, load_func=load_func, id=id)
        sf = wrap_signal_file(
            sf,
            noise=noise,
            target_snr=target_snr,
            wrap_around=wrap_around,
            wrap_around_limit=wrap_around_limit,
        )
        sfs.append(sf)
    return sfs


def load_signal_buffers(
    buffers,
    ids,
    noise=False,
    target_snr=None,
    wrap_around=False,
    wrap_around_limit=None,
):
    sbs = []
    for buf, id in zip(buffers, ids):
        sb = Signal_Buffer(buf, id=id)
        sb = wrap_signal_file(
            sb,
            noise=noise,
            target_snr=target_snr,
            wrap_around=wrap_around,
            wrap_around_limit=wrap_around_limit,
        )
        sbs.append(sb)
    return sbs


def load_controlled_signal_files(target_snr, wrap_around=False, wrap_around_limit=None):
    return load_signal_files(
        "../../data/",
        [
            "controlled_signal.wav",
            "controlled_signal.wav",
            "adversary_controlled_signal.wav",
        ],
        ["legit_signal1", "legit_signal2", "adv_signal"],
        load_func=load_controlled_signal,
        noise=True,
        target_snr=target_snr,
        wrap_around=wrap_around,
        wrap_around_limit=wrap_around_limit,
    )


def load_controlled_signal_buffers(buffers, target_snr=None, noise=False):
    return load_signal_buffers(
        [buffers[0], buffers[1], buffers[2]],
        ids=["legit1", "legit2", "adv"],
        noise=noise,
        target_snr=target_snr,
        wrap_around=True,
        wrap_around_limit=None,
    )


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


def bitstring_to_bytes(s: str) -> bytes:
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")


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


def log_bytes(file_name_stub, byte_list, key_length):
    iteration = 0
    for b in byte_list:
        print(b)
        file_name = file_name_stub + "_" + str(iteration) + ".txt"
        bit_string = bytes_to_bitstring(b, key_length)
        with open(file_name, "w") as file:
            file.write(bit_string)
        iteration += 1


def log_parameters(file_name_stub, name_list, parameter_list):
    csv_file = dict()
    for name, param in zip(name_list, parameter_list):
        csv_file[name] = [param]

    file_name = file_name_stub + "_params.csv"
    df = pd.DataFrame(csv_file)
    df.to_csv(file_name)


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
