import argparse
import os
import sys
from typing import Any, Generator, List, Optional, Tuple

import numpy as np

sys.path.insert(
    1, os.getcwd() + "/../../src/"
)  # Gives us path to Fastzip algorithm in /src
from signal_processing.fastzip import FastZIPProcessing  # noqa: E402

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import bitstring_to_bytes  # noqa: E402

SAMPLING_RATE = 50
MICROPHONE_SAMPLING_RATE = 44100
DATA_DIRECTORY = "./fastzip_data"


def manage_overlapping_chunks(
    new_chunk, previous_chunk
) -> Generator[np.ndarray, None, None]:
    """
    Generate overlapping chunks from a signal buffer.

    :param signal_buffer: Input signal data as a numpy array.
    :param chunk_size: Size of each chunk to be processed.
    :param overlap_size: Number of overlapping samples between consecutive chunks.
    :return: Yields successive overlapping chunks from the signal buffer.
    """

    overlap_size = len(new_chunk)
    return np.concatenate((previous_chunk[-overlap_size:], new_chunk))


def fastzip_event_detection_wrapper_func(
    signal,
    chunk_size,
    overlap_size,
    power_thresh,
    snr_thresh,
    peak_thresh,
    sample_rate,
    peak_status,
    normalize,
    alpha,
):
    events = []
    chunk = signal.read(chunk_size)
    while not signal.get_finished_reading():
        if normalize:
            chunk = FastZIPProcessing.normalize_signal(chunk)
        activity = FastZIPProcessing.activity_filter(
            chunk,
            power_thresh,
            snr_thresh,
            peak_thresh,
            sample_rate,
            peak_status,
            alpha,
        )

        if activity:
            start_timestamp = signal.get_global_index()
            end_timestamp = start_timestamp + chunk_size
            events.append((start_timestamp, end_timestamp))

        if signal.get_finished_reading():
            break
        else:
            new_chunk = signal.read(chunk_size - overlap_size)
            chunk = manage_overlapping_chunks(new_chunk, chunk)

    return events


def fastzip_bit_gen_wrapper(
    chunk, remove_noise, ewma_filter, alpha, bias, n_bits, eqd_delta
):
    if remove_noise:
        chunk = FastZIPProcessing.remove_noise(chunk)
    if ewma_filter:
        chunk = FastZIPProcessing.ewma_filter(abs(chunk), alpha)

    qs_thr = FastZIPProcessing.compute_qs_thr(chunk, bias)

    pts = FastZIPProcessing.generate_equidist_points(
        len(chunk), np.ceil(len(chunk) / n_bits), eqd_delta
    )

    fp = FastZIPProcessing.gen_fp(pts, chunk, qs_thr)

    return fp


def calc_bits(event_file, key_size, *args):
    key = ""
    events, event_sigs = event_file.get_events(1)
    while not event_file.get_finished_reading() and len(key) < key_size:
        chunk = event_sigs[0]
        key += fastzip_bit_gen_wrapper(chunk, *args)
        events, event_sigs = event_file.get_events(1)
    return key[:key_size]


def calc_all_event_bits_fastzip(signals, key_size, *args):
    legit1 = signals[0]
    legit2 = signals[1]
    adv = signals[2]

    legit1_total_bits = []
    legit2_total_bits = []
    adv_total_bits = []

    while not legit1.get_finished_reading() and not legit2.get_finished_reading():
        legit1_bits = calc_bits(legit1, key_size, *args)
        legit2_bits = calc_bits(legit2, key_size, *args)

        if not adv.get_finished_reading():
            adv_bits = calc_bits(adv, key_size, *args)
            adv_total_bits.append(adv_bits)

        legit1_total_bits.append(legit1_bits)
        legit2_total_bits.append(legit2_bits)

        if not legit1.get_finished_reading() and not legit2.get_finished_reading():
            legit2.sync(legit1)

            if not adv.get_finished_reading():
                adv.sync(legit1)

    return legit1_total_bits, legit2_total_bits, adv_total_bits


def fastzip_wrapper_function(
    sensor,
    n_bits: int,
    chunk_size: int,
    overlap_size: int,
    power_thr: float,
    snr_thr: float,
    peak_thr: int,
    bias: int,
    eqd_delta: int,
    sampling_rate: int,
    key_length: int,
    peak_status: Optional[bool] = None,
    ewma_filter: Optional[float] = None,
    alpha: Optional[float] = None,
    remove_noise: Optional[bool] = None,
    normalize: Optional[bool] = None,
) -> List[int]:

    accumulated_bits = ""
    number_of_bits = 0
    samples_read = chunk_size
    chunk = sensor.read(chunk_size)
    while not sensor.get_finished_reading() and number_of_bits < key_length:
        bits = FastZIPProcessing.fastzip_algo(
            [chunk],
            [n_bits],
            [power_thr],
            [snr_thr],
            [peak_thr],
            [bias],
            [sampling_rate],
            [eqd_delta],
            [peak_status],
            [ewma_filter],
            [alpha],
            [remove_noise],
            [normalize],
            return_bitstring=True,
        )

        if bits:
            accumulated_bits += bits
            number_of_bits += n_bits

        new_overlap_chunk = sensor.read(chunk_size - overlap_size)
        chunk = manage_overlapping_chunks(new_overlap_chunk, chunk)
        samples_read += overlap_size
    if len(accumulated_bits) >= key_length:
        accumulated_bits = accumulated_bits[:key_length]
    return bitstring_to_bytes(accumulated_bits), samples_read


def golden_signal(sample_num: int) -> np.ndarray:
    """
    Generate a golden signal using a predefined seed.

    :param sample_num: Number of samples in the generated signal.
    :return: Golden signal as a numpy array.
    """
    np.random.seed(0)
    output = np.random.rand(sample_num)
    return output


def adversary_signal(sample_num: int) -> np.ndarray:
    """
    Generate an adversary signal using a predefined seed.

    :param sample_num: Number of samples in the generated signal.
    :return: Adversary signal as a numpy array.
    """
    np.random.seed(12)
    output = np.random.rand(sample_num)
    return output


def parse_command_line_args(
    window_size_default: int = 200,
    overlap_size_default: int = 100,
    buffer_size_default: int = 50000,
    n_bits_default: int = 18,
    key_length_default: int = 128,
    bias_default: int = 0,
    eqd_delta_default: int = 1,
    peak_status_default: Any | None = None,
    ewma_filter_default: Any | None = None,
    alpha_default: Any | None = None,
    remove_noise_default: Any | None = None,
    normalize_default: bool = True,
    power_threshold_default: int = 70,
    snr_threshold_default: float = 1.2,
    number_peaks_default: int = 0,
    target_snr_default: int = 20,
    trials_default: int = 1000,
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    Optional[bool],
    Optional[float],
    Optional[float],
    Optional[bool],
    Optional[bool],
    float,
    float,
    int,
    int,
    int,
]:
    """
    Parse command line arguments for the application using argparse.

    :return: A tuple containing all parsed values.
    """
    parser = argparse.ArgumentParser()

    # Add arguments with descriptions
    parser.add_argument("-ws", "--window-size", type=int, default=window_size_default)
    parser.add_argument("-os", "--overlap-size", type=int, default=overlap_size_default)
    parser.add_argument("-bs", "--buffer-size", type=int, default=buffer_size_default)
    parser.add_argument("-nb", "--n-bits", type=int, default=n_bits_default)
    parser.add_argument("-kl", "--key-length", type=int, default=key_length_default)
    parser.add_argument("-b", "--bias", type=int, default=bias_default)
    parser.add_argument("-ed", "--eqd-delta", type=int, default=eqd_delta_default)
    parser.add_argument("-ps", "--peak-status", type=bool, default=peak_status_default)
    parser.add_argument("-ef", "--ewma-filter", type=bool, default=ewma_filter_default)
    parser.add_argument("-a", "--alpha", type=float, default=alpha_default)
    parser.add_argument(
        "-rn", "--remove-noise", type=bool, default=remove_noise_default
    )
    parser.add_argument("-n", "--normalize", type=bool, default=normalize_default)
    parser.add_argument(
        "-pt", "--power-threshold", type=int, default=power_threshold_default
    )
    parser.add_argument(
        "-st", "--snr-threshold", type=float, default=snr_threshold_default
    )
    parser.add_argument("-np", "--number-peaks", type=int, default=number_peaks_default)
    parser.add_argument("-snr", "--snr-level", type=int, default=target_snr_default)
    parser.add_argument("-t", "--trials", type=int, default=trials_default)

    # Parsing command-line arguments
    args = parser.parse_args()

    # Extracting arguments
    window_size = getattr(args, "window_size")
    overlap_size = getattr(args, "overlap_size")
    buffer_size = getattr(args, "buffer_size")
    n_bits = getattr(args, "n_bits")
    key_length = getattr(args, "key_length")
    bias = getattr(args, "bias")
    eqd_delta = getattr(args, "eqd_delta")
    peak_status = getattr(args, "peak_status")
    ewma_filter = getattr(args, "ewma_filter")
    alpha = getattr(args, "alpha")
    remove_noise = getattr(args, "remove_noise")
    normalize = getattr(args, "normalize")
    power_threshold = getattr(args, "power_threshold")
    snr_threshold = getattr(args, "snr_threshold")
    number_peaks = getattr(args, "number_peaks")
    snr_level = getattr(args, "snr_level")
    trials = getattr(args, "trials")

    return (
        window_size,
        overlap_size,
        buffer_size,
        n_bits,
        key_length,
        bias,
        eqd_delta,
        peak_status,
        ewma_filter,
        alpha,
        remove_noise,
        normalize,
        power_threshold,
        snr_threshold,
        number_peaks,
        snr_level,
        trials,
    )
