import os
import sys
import unittest

import numpy as np

sys.path.insert(1, os.path.join(os.getcwd(), "..", "src", "signal_processing"))
from fastzip import FastZIPProcessing  # noqa: E402


class TestFastZIPProcessing(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 100  # Hz
        self.data = np.random.normal(
            0, 1, self.sample_rate
        )  # Simulate 1-second of data
        self.alpha = 0.015
        self.power_threshold = -12
        self.snr_threshold = 1.2
        self.eqd_delta = 1
        self.bias = 0

    def test_normalize_signal(self):
        normalized_data = FastZIPProcessing.normalize_signal(self.data)
        self.assertEqual(len(normalized_data), len(self.data))
        self.assertAlmostEqual(np.mean(normalized_data), 0)
        print("Test Normalize Signal: PASSED")

    def test_remove_noise(self):
        noise_removed_data = FastZIPProcessing.remove_noise(self.data)
        self.assertEqual(len(noise_removed_data), len(self.data))
        print("Test Remove Noise: PASSED")

    def test_ewma_filter(self):
        filtered_data = FastZIPProcessing.ewma_filter(self.data, self.alpha)
        self.assertEqual(len(filtered_data), len(self.data))
        print("Test EWMA Filter: PASSED")

    def test_compute_sig_power(self):
        power = FastZIPProcessing.compute_sig_power(self.data)
        self.assertIsNotNone(power)
        print("Test Compute Signal Power: PASSED")

    def test_compute_snr(self):
        snr = FastZIPProcessing.compute_snr(self.data)
        self.assertIsNotNone(snr)
        print("Test Compute SNR: PASSED")

    def test_get_peaks(self):
        peaks = FastZIPProcessing.get_peaks(self.data, self.sample_rate)
        self.assertIsInstance(peaks, int)
        print("Test Get Peaks: PASSED")

    def test_activity_filter(self):
        activity = FastZIPProcessing.activity_filter(
            self.data,
            self.power_threshold,
            self.snr_threshold,
            0,
            self.sample_rate,
            True,
            self.alpha,
        )
        self.assertIn(activity, [True, False])
        print("Test Activity Filter: PASSED")

    def test_compute_qs_thr(self):
        threshold = FastZIPProcessing.compute_qs_thr(self.data, self.bias)
        self.assertIsNotNone(threshold)
        print("Test Compute Quantization Threshold: PASSED")

    def test_generate_equidist_points(self):
        points = FastZIPProcessing.generate_equidist_points(
            len(self.data), 10, self.eqd_delta
        )
        self.assertIsInstance(points, list)
        print("Test Generate Equidistant Points: PASSED")

    def test_compute_fingerprint(self):
        fingerprint = FastZIPProcessing.compute_fingerprint(
            self.data,
            12,
            self.power_threshold,
            self.snr_threshold,
            0,
            self.bias,
            self.sample_rate,
            self.eqd_delta,
            False,
            False,
            self.alpha,
            False,
            True,
        )
        self.assertIsInstance(fingerprint, str)
        print("Test Compute Fingerprint: PASSED")


if __name__ == "__main__":
    unittest.main()
