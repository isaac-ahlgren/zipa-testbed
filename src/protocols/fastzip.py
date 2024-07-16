from math import ceil

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from protocols.protocol_interface import ProtocolInterface


# WIP
class FastZIP_Protocol(ProtocolInterface):
    """
    An implementation of a cryptographic protocol that uses signal processing to secure
    communication between devices. It includes methods for normalizing signals, removing noise,
    and detecting specific patterns or features within signals.

    :param parameters: Configuration parameters for the protocol.
    :param sensor: The sensor object to collect data from.
    :param logger: Logger object for recording protocol operations.
    """

    def __init__(self, parameters, sensor, logger):
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "FastZIP_Protocol"
        self.wip = True
        self.count = 0

    def extract_context(self):
        pass

    def parameters(self, is_host):
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"window_length: {self.window_len}\n"
        parameters += f"band_length: {self.band_len}\n"
        parameters += f"time_length: {self.time_length}\n"

    def device_protocol(self, host):
        pass

    def host_protocol_single_threaded(self, device_socket):
        pass

    def normalize_signal(sig):
        """
        Normalizes the signal by subtracting its mean.

        :param sig: The signal to be normalized.
        :returns: The normalized signal.
        """
        # Check if sig is non-zero
        if len(sig) == 0:
            print("normalize_signal: signal must have non-zero length!")
            return

        # Noramlized signal to be returned
        norm_sig = np.copy(sig)

        # Subtract mean from sig
        norm_sig = norm_sig - np.mean(norm_sig)

        return norm_sig

    def remove_noise(data):
        """
        Applies noise removal techniques to the data, including Gaussian and Savitzky-Golay filters.

        :param data: The data from which noise is to be removed.
        :returns: The data after noise removal.
        """
        rn_data = np.zeros(data.shape)
        rn_data = gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)

        return rn_data

    def ewma_filter(data, alpha=0.15):
        """
        Applies an Exponentially Weighted Moving Average (EWMA) filter to the data.

        :param data: The data to filter.
        :param alpha: The decay factor for the EWMA filter.
        :returns: The filtered data.
        """
        ewma_data = np.zeros(len(data))
        ewma_data[0] = data[0]
        for i in range(1, len(ewma_data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        return ewma_data

    def compute_sig_power(sig):
        """
        Computes the power of the signal.

        :param sig: The signal whose power is to be computed.
        :returns: The power of the signal in dB.
        """
        if len(sig) == 0:
            print("compute_sig_power: signal must have non-zero length!")
            return
        return 10 * np.log10(np.sum(sig**2) / len(sig))

    def compute_snr(sig):
        """
        Computes the Signal-to-Noise Ratio (SNR) of the signal.

        :param sig: The signal whose SNR is to be computed.
        :returns: The SNR of the signal.
        """
        if len(sig) == 0:
            print("compute_snr: signal must have non-zero length!")
            return
        return np.mean(abs(sig)) / np.std(abs(sig))

    def get_peaks(sig):
        """
        Identifies peaks in the signal based on average peak height.

        :param sig: The signal to analyze.
        :param sample_rate: The sample rate of the signal.
        :returns: The number of peaks detected in the signal.
        """
        peak_height = np.mean(sorted(sig)[-9:]) * 0.2
        peaks, _ = FastZIP_Protocol.find_peaks(
            sig, height=peak_height, distance=0.25 * self.sample_rate
        )
        return len(peaks)

    def activity_filter(signal, power_thresh, snr_thresh, peak_thresh):
        """
        Filters signals based on power, SNR, and peak thresholds to detect activity.

        :param signal: The signal to filter.
        :param power_thresh: The power threshold for detecting activity.
        :param snr_thresh: The SNR threshold for detecting activity.
        :param peak_thresh: The peak threshold for detecting activity.
        :returns: True if activity is detected, False otherwise.
        """
        # Ensure signal is a numpy array
        signal = np.copy(signal)

        # Initialize metrics
        power, snr, n_peaks = 0, 0, 0

        # Compute signal power (similar for all sensor types
        power = FastZIP_Protocol.compute_sig_power(signal)

        # Find peaks
        peaks = FastZIP_Protocol.get_peaks(signal)

        # Compute signal's SNR
        snr = FastZIP_Protocol.compute_snr(signal)

        # Check against thresholds to determine if activity is present
        activity_detected = False
        if power > power_thresh and snr > snr_thresh and peaks > peak_thresh:
            activity_detected = True

        return activity_detected

    def compute_qs_thr(chunk, bias):
        """
        Computes a threshold for quantizing signals based on the median and a bias.

        :param chunk: The data chunk to compute the threshold for.
        :param bias: The bias to add to the median to compute the threshold.
        :returns: The computed threshold.
        """
        # Make a copy of chunk
        chunk_cpy = np.copy(chunk)

        # Sort the chunk
        chunk_cpy.sort()

        return np.median(chunk_cpy) + bias

    def generate_equidist_points(self, chunk_len, step, eqd_delta):
        """
        Generates equidistant points within a data chunk.

        :param chunk_len: The length of the data chunk.
        :param step: The step size for generating points.
        :param eqd_delta: The equidistant delta for point generation.
        :returns: A list of equidistant points and the count of such points.
        """
        # Equidistant delta cannot be bigger than the step
        if eqd_delta > step:
            print('generate_equidist_points: "eqd_delta" must be smaller than "step"')
            return -1, 0

        # Store equidistant points
        eqd_points = []

        # Generate equdistant points
        for i in range(0, ceil(chunk_len / eqd_delta)):
            eqd_rand_points.append(
                np.arange(
                    0 + eqd_delta * i,
                    chunk_len + eqd_delta * i,
                    ceil(chunk_len / n_bits),
                )
                % chunk_len
            )

        return eqd_rand_points, len(eqd_rand_points)

    def compute_fingerprint(
        data,
        n_bits,
        power_thresh,
        snr_thresh,
        peak_thresh,
        bias,
        ewma_filter=False,
        alpha=0.015,
        remove_noise=False,
        normalize=False,
    ):
        """
        Computes a binary fingerprint from sensor data using specified thresholds and filters.

        :param data: The sensor data to process.
        :param n_bits: The number of bits to generate for the fingerprint.
        :param power_thresh: The power threshold for detecting activity.
        :param snr_thresh: The SNR threshold for detecting activity.
        :param peak_thresh: The peak threshold for detecting activity.
        :param bias: The bias to use in quantization.
        :param ewma_filter: Whether to apply an EWMA filter.
        :param alpha: The alpha value for the EWMA filter.
        :param remove_noise: Whether to remove noise from the data.
        :param normalize: Whether to normalize the data.
        :returns: A binary fingerprint of the data.
        """
        fp = None

        if normalize:
            chunk = normalize_signal(chunk)

        activity = FastZIP_Protocol.activity_filter(
            chunk, power_thresh, snr_thresh, peak_thresh
        )
        if activity:
            if remove_noise:
                chunk = FastZIP_Protocol.remove_noise(chunk)

            if ewma_filter:
                chunk = FastZIP_Protocol.ewma_filter(abs(chunk), alpha=alpha)

            qs_thr = FastZIP_Protocol.compute_qs_thr(chunk, bias)

            pts = FastZIP_Protocol.generate_equidist_points(
                len(data), n_bits, eqd_delta
            )

            fp = ""
            for pt in pts:
                if data[pt] > qs_thr:
                    fp += "1"
                else:
                    fp += "0"

        return fp

    def fastzip_algo(
        sensor_data_list,
        n_bits_list,
        power_thresh_list,
        snr_thresh_list,
        peak_thresh_list,
        bias_list,
        ewma_filter_list=None,
        alpha_list=None,
        remove_noise_list=None,
        normalize_list=None,
    ):
        """
        Aggregates the fingerprint generation for multiple sensors and configurations,
        producing a single cryptographic key from multiple data sources.

        :param sensor_data_list: List of sensor data arrays from which fingerprints are derived.
        :param n_bits_list: List of integers representing the number of bits to generate from each data set.
        :param power_thresh_list: List of power thresholds for activity detection in each data set.
        :param snr_thresh_list: List of SNR thresholds for activity detection in each data set.
        :param peak_thresh_list: List of peak count thresholds for activity detection in each data set.
        :param bias_list: List of biases used in the quantization step for each data set.
        :param ewma_filter_list: Optional list of booleans indicating whether to apply EWMA filtering to each data set.
        :param alpha_list: Optional list of alpha values for the EWMA filter for each data set.
        :param remove_noise_list: Optional list of booleans indicating whether to apply noise removal to each data set.
        :param normalize_list: Optional list of booleans indicating whether to normalize each data set before processing.
        :returns: A concatenated string of '0' and '1' representing the cryptographic key derived from all data sets.
        """
        key = ""

        for i in range(len(sensor_data)):
            data = sensor_data_list[i]
            n_bits = n_bits_list[i]
            power_thresh = power_thresh_list[i]
            snr_thresh = snr_thresh_list[i]
            peak_thresh = peak_thresh_list[i]
            bias = bias_list[i]

            if ewma_filter_list == None:
                ewma_filter = False
            else:
                ewma_filter = ewma_filter_list[i]

            if alpha_list == None:
                alpha = 0.015
            else:
                alpha = alpha_list[i]

            if remove_noise_list == None:
                remove_noise = False
            else:
                remove_noise = remove_noise_list[i]

            if normalize_list == None:
                normalize = False
            else:
                normalize = normalize[i]

            bits = FastZIP_Protocol.compute_fingerprint(
                data,
                n_bits,
                power_thresh,
                snr_thresh,
                peak_thresh,
                bias,
                ewma_filter=ewma_filter,
                alpha=alpha,
                remove_noise=remove_noise,
                normalize=normalize,
            )

            if bits != None:
                key += bits
        return key


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

    fastzip = FastZIP_Protocol(sample_rate)

    bar_binary_data = fastzip.signal_to_bits(bar_signal, sensor_type="bar")
    print("Barometer binary data:", bar_binary_data)
    acc_binary_data = fastzip.signal_to_bits(acc_signal, sensor_type="acc")
    print("Accelerometer binary data:", acc_binary_data)
    gyr_binary_data = fastzip.signal_to_bits(gyr_signal, sensor_type="gyrW")
    print("Gyroscope binary data:", gyr_binary_data)
