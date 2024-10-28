from math import ceil
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter


class FastZIPProcessing:
    def fastzip_algo(
        sensor_data_list: List[np.ndarray],
        n_bits_list: List[int],
        power_thresh_list: List[float],
        snr_thresh_list: List[float],
        peak_thresh_list: List[int],
        bias_list: List[float],
        sample_rate_list: List[int],
        eqd_delta_list: List[int],
        peak_status_list: Optional[List[bool]] = None,
        ewma_filter_list: Optional[List[bool]] = None,
        alpha_list: Optional[List[float]] = None,
        remove_noise_list: Optional[List[bool]] = None,
        normalize_list: Optional[List[bool]] = None,
        return_bitstring=False,
    ) -> bytes:
        """
        Main algorithm for processing sensor data and generating a cryptographic key.

        :param sensor_data_list: List of sensor data arrays.
        :param n_bits_list: List of numbers specifying how many bits to extract.
        :param power_thresh_list: List of power thresholds for determining activity.
        :param snr_thresh_list: List of SNR thresholds for signal filtering.
        :param peak_thresh_list: List of peak thresholds for peak detection.
        :param bias_list: List of biases to adjust the thresholds.
        :param sample_rate_list: List of sample rates of the input data.
        :param eqd_delta_list: List of deltas for equidistant point calculation.
        :param peak_status_list: Optional list indicating whether peak status is considered.
        :param ewma_filter_list: Optional list indicating whether to apply EWMA filtering.
        :param alpha_list: Optional list of alpha values for EWMA calculation.
        :param remove_noise_list: Optional list indicating whether to apply noise removal.
        :param normalize_list: Optional list indicating whether to normalize the data.
        :return: A bytes object representing the generated cryptographic key.
        """

        def bitstring_to_bytes(s: str) -> bytes:
            if s == "":
                return b""
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        key = ""

        for i in range(len(sensor_data_list)):
            data = sensor_data_list[i]
            n_bits = n_bits_list[i]
            power_thresh = power_thresh_list[i]
            snr_thresh = snr_thresh_list[i]
            peak_thresh = peak_thresh_list[i]
            bias = bias_list[i]
            sample_rate = sample_rate_list[i]
            eqd_delta = eqd_delta_list[i]

            if peak_status_list is None:
                peak_status = False
            else:
                peak_status = peak_status_list[i]

            if ewma_filter_list is None:
                ewma_filter = False
            else:
                ewma_filter = ewma_filter_list[i]

            if alpha_list is None or alpha_list[i] is None:
                alpha = 0.015
            else:
                alpha = alpha_list[i]

            if remove_noise_list is None:
                remove_noise = False
            else:
                remove_noise = remove_noise_list[i]

            if normalize_list is None:
                normalize = False
            else:
                normalize = normalize_list[i]

            bits = FastZIPProcessing.compute_fingerprint(
                data,
                n_bits,
                power_thresh,
                snr_thresh,
                peak_thresh,
                bias,
                sample_rate,
                eqd_delta,
                peak_status=peak_status,
                ewma_filter=ewma_filter,
                alpha=alpha,
                remove_noise=remove_noise,
                normalize=normalize,
            )

            if bits is not None:
                key += bits
        if return_bitstring is True:
            return key
        else:
            return bitstring_to_bytes(key)

    def normalize_signal(sig: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize a signal by subtracting its mean.

        :param sig: Input signal array.
        :return: Normalized signal array, or None if the input signal is empty.
        """
        if len(sig) == 0:
            print("normalize_signal: signal must have non-zero length!")
            return

        norm_sig = np.copy(sig)

        norm_sig = norm_sig - np.mean(norm_sig)

        return norm_sig

    def remove_noise(data: np.ndarray) -> np.ndarray:
        """
        Apply a Gaussian filter after a Savitzky-Golay filter to remove noise from the data.

        :param data: Input signal data.
        :return: Noise-reduced signal.
        """
        rn_data = np.zeros(data.shape)
        rn_data = gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)

        return rn_data

    def ewma_filter(data: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply an Exponentially Weighted Moving Average (EWMA) filter to the data.

        :param data: Input signal data.
        :param alpha: Smoothing factor used in the EWMA calculation.
        :return: Filtered signal.
        """
        ewma_data = np.zeros(len(data))
        ewma_data[0] = data[0]
        for i in range(1, len(ewma_data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        return ewma_data

    def compute_sig_power(sig: np.ndarray) -> Optional[float]:
        """
        Compute the power of a signal.

        :param sig: Signal array.
        :return: Power of the signal in dB, or None if the signal is empty.
        """
        if len(sig) == 0:
            print("compute_sig_power: signal must have non-zero length!")
            return
        rms_sqr = np.mean(sig**2)
        if rms_sqr == 0:
            return 0
        else:
            return 10 * np.log10(rms_sqr)

    def compute_snr(sig: np.ndarray) -> Optional[float]:
        """
        Compute the Signal-to-Noise Ratio (SNR) of a signal.

        :param sig: Signal array.
        :return: SNR of the signal, or None if the signal is empty.
        """
        if len(sig) == 0:
            print("compute_snr: signal must have non-zero length!")
            return
        std = np.std(abs(sig))
        if std == 0:
            return 0
        else:
            return np.mean(abs(sig)) / np.std(abs(sig))

    def get_peaks(sig: np.ndarray, sample_rate: int) -> int:
        """
        Identify peaks in a signal based on the average height and a specified sample rate.

        :param sig: Signal array.
        :param sample_rate: Sampling rate of the signal.
        :return: Number of detected peaks.
        """
        peak_height = np.mean(sorted(sig)[-9:]) * 0.2
        peaks, _ = find_peaks(sig, height=peak_height, distance=0.25 * sample_rate)
        return len(peaks)

    def activity_filter(
        signal: np.ndarray,
        power_thresh: float,
        snr_thresh: float,
        peak_thresh: int,
        sample_rate: int,
        peak_status: bool,
        alpha: float,
    ) -> bool:
        """
        Filter signal activity based on power, SNR, and peak detection thresholds.

        :param signal: Input signal array.
        :param power_thresh: Power threshold.
        :param snr_thresh: SNR threshold.
        :param peak_thresh: Peak count threshold.
        :param sample_rate: Sampling rate of the signal.
        :param peak_status: Boolean flag to consider peak status.
        :param alpha: EWMA alpha value.
        :return: True if the activity is detected, otherwise False.
        """
        signal = np.copy(signal)

        power, snr, peaks = 0, 0, 0

        power = FastZIPProcessing.compute_sig_power(signal)

        if peak_status:
            abs_signal = FastZIPProcessing.ewma_filter(abs(signal), alpha)
            peaks = FastZIPProcessing.get_peaks(abs_signal, sample_rate)

        snr = FastZIPProcessing.compute_snr(signal)

        activity_detected = False
        if power > power_thresh and snr > snr_thresh and peaks >= peak_thresh:
            activity_detected = True

        return activity_detected

    def compute_qs_thr(chunk: np.ndarray, bias: float) -> float:
        """
        Compute the threshold for quiescent state determination in a signal chunk.

        :param chunk: Input signal chunk.
        :param bias: Bias to adjust the threshold.
        :return: Computed threshold.
        """
        chunk_cpy = np.copy(chunk)

        chunk_cpy.sort()

        return np.median(chunk_cpy) + bias

    def generate_equidist_points(
        chunk_len: int, step: int, eqd_delta: int
    ) -> List[np.ndarray]:
        """
        Generate equidistant points within a signal chunk.

        :param chunk_len: Length of the chunk.
        :param step: Step size between points.
        :param eqd_delta: Delta for equidistant calculation.
        :return: List of arrays containing equidistant points.
        """
        if eqd_delta > step:
            print('generate_equidist_points: "eqd_delta" must be smaller than "step"')
            return -1, 0

        eqd_rand_points = []

        for i in range(0, ceil(chunk_len / eqd_delta)):
            eqd_rand_points.append(
                np.arange(
                    0 + eqd_delta * i,
                    chunk_len + eqd_delta * i,
                    step,
                )
                % chunk_len
            )

        return eqd_rand_points

    def compute_fingerprint(
        data: np.ndarray,
        n_bits: int,
        power_thresh: float,
        snr_thresh: float,
        peak_thresh: int,
        bias: float,
        sample_rate: int,
        eqd_delta: int,
        peak_status: bool = False,
        ewma_filter: bool = False,
        alpha: float = 0.015,
        remove_noise: bool = False,
        normalize: bool = False,
    ) -> Optional[str]:
        """
        Processes a single sensor data array to generate a fingerprint string based on specified thresholds and settings.

        :param data: Sensor data as a NumPy array.
        :param n_bits: Number of bits to extract.
        :param power_thresh: Power threshold for determining activity.
        :param snr_thresh: SNR threshold for signal filtering.
        :param peak_thresh: Peak threshold for peak detection.
        :param bias: Bias to adjust the threshold.
        :param sample_rate: Sample rate of the input data.
        :param eqd_delta: Delta for equidistant point calculation.
        :param peak_status: Indicates whether peak status is considered.
        :param ewma_filter: Indicates whether to apply EWMA filtering.
        :param alpha: Alpha value for EWMA calculation.
        :param remove_noise: Indicates whether to apply noise removal.
        :param normalize: Indicates whether to normalize the data.
        :return: A string representing the fingerprint, or None if no activity is detected.
        """
        chunk = np.copy(data)
        fp = None

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
            if remove_noise:
                chunk = FastZIPProcessing.remove_noise(chunk)
            if ewma_filter:
                chunk = FastZIPProcessing.ewma_filter(abs(chunk), alpha)

            qs_thr = FastZIPProcessing.compute_qs_thr(chunk, bias)

            pts = FastZIPProcessing.generate_equidist_points(
                len(chunk), ceil(len(chunk) / n_bits), eqd_delta
            )

            fp = ""
            for pt in pts:
                for index in pt:
                    if chunk[int(index)] > qs_thr:
                        fp += "1"
                    else:
                        fp += "0"

        return fp


# Example usage:
if __name__ == "__main__":
    sample_rate = 100  # Hz, replace with actual sample rate

    # Simulate 1D barometer data (pressure readings)
    bar_signal = np.random.normal(1013, 5, size=sample_rate * 60)  # 1 minute of data
    # Example accelerometer data; replace with actual sensor data
    acc_signal = np.random.normal(
        0, 1, size=(sample_rate * 60, 3)
    )  # Assuming 3-axis accelerometer data for 1 minute
    # Simulate 3D gyroscope data (angular velocity)
    gyr_signal = np.random.normal(
        0, 1, size=(sample_rate * 60, 3)
    )  # 1 minute of 3-axis data

    fastzip = FastZIPProcessing(sample_rate)

    bar_binary_data = fastzip.signal_to_bits(bar_signal, sensor_type="bar")
    print("Barometer binary data:", bar_binary_data)
    acc_binary_data = fastzip.signal_to_bits(acc_signal, sensor_type="acc")
    print("Accelerometer binary data:", acc_binary_data)
    gyr_binary_data = fastzip.signal_to_bits(gyr_signal, sensor_type="gyrW")
    print("Gyroscope binary data:", gyr_binary_data)
