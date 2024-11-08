import os
import sys

import numpy as np

sys.path.insert(1, os.path.join(os.getcwd(), "src", "signal_processing"))
from fastzip import FastZIPProcessing  # noqa: E402


def setUp():
    sample_rate = 100  # Hz
    data = np.random.normal(0, 1, sample_rate)  # Simulate 1-second of data
    return sample_rate, data


def test_normalize_signal():
    sample_rate, data = setUp()
    normalized_data = FastZIPProcessing.normalize_signal(data)
    assert len(normalized_data) == len(data)  # nosec
    assert np.isclose(np.mean(normalized_data), 0)  # nosec


def test_remove_noise():
    sample_rate, data = setUp()
    noise_removed_data = FastZIPProcessing.remove_noise(data)
    assert len(noise_removed_data) == len(data)  # nosec


def test_ewma_filter():
    sample_rate, data = setUp()
    alpha = 0.015
    filtered_data = FastZIPProcessing.ewma_filter(data, alpha)
    assert len(filtered_data) == len(data)  # nosec


def test_compute_sig_power():
    sample_rate, data = setUp()
    power = FastZIPProcessing.compute_sig_power(data)
    assert power is not None  # nosec


def test_compute_snr():
    sample_rate, data = setUp()
    snr = FastZIPProcessing.compute_snr(data)
    assert snr is not None  # nosec


def test_get_peaks():
    sample_rate, data = setUp()
    peaks = FastZIPProcessing.get_peaks(data, sample_rate)
    assert isinstance(peaks, int)  # nosec


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
    assert activity in [True, False]  # nosec


def test_compute_qs_thr():
    sample_rate, data = setUp()
    bias = 0
    threshold = FastZIPProcessing.compute_qs_thr(data, bias)
    assert threshold is not None  # nosec


def test_generate_equidist_points():
    def test_func(chunk_len: int, step: int, eqd_delta: int):
        eqd_rand_points = []

        for i in range(0, np.ceil(chunk_len / eqd_delta).astype(int)):
            eqd_rand_points.append(
                np.arange(
                    0 + eqd_delta * i,
                    chunk_len + eqd_delta * i,
                    step,
                )
                % chunk_len
            )

        return eqd_rand_points

    sample_rate, data = setUp()
    eqd_delta = 1
    test_points = test_func(len(data), 10, eqd_delta)
    points = FastZIPProcessing.generate_equidist_points(len(data), 10, eqd_delta)
    assert np.array_equal(points, test_points)  # nosec


def test_gen_fp():
    def test_func(chunk, eqd_delta, step, qs_thr):
        eqd_rand_points = []

        for i in range(0, np.ceil(len(chunk) / eqd_delta).astype(int)):
            eqd_rand_points.append(
                np.arange(
                    0 + eqd_delta * i,
                    len(chunk) + eqd_delta * i,
                    step,
                )
                % len(chunk)
            )

        pts = eqd_rand_points

        fp = ""
        for pt in pts:
            for index in pt:
                if chunk[int(index)] > qs_thr:
                    fp += "1"
                else:
                    fp += "0"
        return fp

    sample_rate, data = setUp()

    eqd_delta = 1
    ref = test_func(data, eqd_delta, 10, 0.5)
    pts = FastZIPProcessing.generate_equidist_points(len(data), 10, eqd_delta)
    fp = FastZIPProcessing.gen_fp(pts, data, 0.5)
    assert ref == fp


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
    assert isinstance(fingerprint, str)  # nosec
