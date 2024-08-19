import os
import sys
import unittest
import numpy as np


sys.path.insert(1, os.path.join(os.getcwd(), "src", "signal_processing"))
from fastzip import FastZIPProcessing  # noqa: E402


def setUp():
    sample_rate = 100  # Hz
    data = np.random.normal(
        0, 1, sample_rate
    )  # Simulate 1-second of data
    return sample_rate, data


def test_normalize_signal():
    sample_rate, data = setUp()
    normalized_data = FastZIPProcessing.normalize_signal(data)
    assert len(normalized_data) == len(data)
    assert np.isclose(np.mean(normalized_data), 0)


def test_remove_noise():
    sample_rate, data = setUp()
    noise_removed_data = FastZIPProcessing.remove_noise(data)
    assert len(noise_removed_data) == len(data)


def test_ewma_filter():
    sample_rate, data = setUp()
    alpha = 0.015
    filtered_data = FastZIPProcessing.ewma_filter(data, alpha)
    assert len(filtered_data) == len(data)


def test_compute_sig_power():
    sample_rate, data = setUp()
    power = FastZIPProcessing.compute_sig_power(data)
    assert power is not None


def test_compute_snr():
    sample_rate, data = setUp()
    snr = FastZIPProcessing.compute_snr(data)
    assert snr is not None


def test_get_peaks():
    sample_rate, data = setUp()
    peaks = FastZIPProcessing.get_peaks(data, sample_rate)
    assert isinstance(peaks, int)


def test_activity_filter():
    sample_rate, data = setUp()
    power_threshold = -12
    snr_threshold = 1.2
    alpha = 0.015
    activity = FastZIPProcessing.activity_filter(
        data,
        power_threshold,
        snr_threshold,
        0,
        sample_rate,
        True,
        alpha,
    )
    assert activity in [True, False]


def test_compute_qs_thr():
    sample_rate, data = setUp()
    bias = 0
    threshold = FastZIPProcessing.compute_qs_thr(data, bias)
    assert threshold is not None


def test_generate_equidist_points():
    sample_rate, data = setUp()
    eqd_delta = 1
    points = FastZIPProcessing.generate_equidist_points(
        len(data), 10, eqd_delta
    )
    assert isinstance(points, list)


def test_compute_fingerprint():
    sample_rate, data = setUp()
    power_threshold = -12
    snr_threshold = 1.2
    eqd_delta = 1
    bias = 0
    alpha = 0.015
    fingerprint = FastZIPProcessing.compute_fingerprint(
        data,
        12,
        power_threshold,
        snr_threshold,
        0,
        bias,
        sample_rate,
        eqd_delta,
        False,
        False,
        alpha,
        False,
        True,
    )
    assert isinstance(fingerprint, str)


