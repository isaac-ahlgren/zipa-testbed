from math import ceil

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter

from protocols.protocol_interface import ProtocolInterface


# WIP
class FastZIP_Protocol(ProtocolInterface):
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
        rn_data = np.zeros(data.shape)
        rn_data = gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)

        return rn_data

    def ewma_filter(data, alpha=0.15):
        ewma_data = np.zeros(len(data))
        ewma_data[0] = data[0]
        for i in range(1, len(ewma_data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        return ewma_data

    def compute_sig_power(sig):
        if len(sig) == 0:
            print("compute_sig_power: signal must have non-zero length!")
            return
        return 10 * np.log10(np.sum(sig**2) / len(sig))

    def compute_snr(sig):
        if len(sig) == 0:
            print("compute_snr: signal must have non-zero length!")
            return
        return np.mean(abs(sig)) / np.std(abs(sig))

    def get_peaks(sig, sample_rate):
        peak_height = np.mean(sorted(sig)[-9:]) * 0.2
        peaks, _ = find_peaks(sig, height=peak_height, distance=0.25 * sample_rate)
        return len(peaks)

    def activity_filter(signal, power_thresh, snr_thresh, peak_thresh, sample_rate):
        # Ensure signal is a numpy array
        signal = np.copy(signal)

        # Initialize metrics
        power, snr, n_peaks = 0, 0, 0

        # Compute signal power (similar for all sensor types
        power = FastZIP_Protocol.compute_sig_power(signal)
        print("Power threshold: ", power_thresh)
        print("Signal Power: ", power)

        # Find peaks
        peaks = FastZIP_Protocol.get_peaks(signal, sample_rate)
        print("Peak threshold: ", peak_thresh)
        print("Peaks: ", peaks)

        # Compute signal's SNR
        snr = FastZIP_Protocol.compute_snr(signal)
        print("SNR Threshold: ", snr_thresh)
        print("Signal SNR: ", snr)

        # Check against thresholds to determine if activity is present
        activity_detected = False
        if power > power_thresh and snr > snr_thresh and peaks > peak_thresh:
            activity_detected = True

        return activity_detected

    def compute_qs_thr(chunk, bias):
        # Make a copy of chunk
        chunk_cpy = np.copy(chunk)

        # Sort the chunk
        chunk_cpy.sort()

        return np.median(chunk_cpy) + bias

    def generate_equidist_points(chunk_len, step, eqd_delta):
        # Equidistant delta cannot be bigger than the step
        if eqd_delta > step:
            print('generate_equidist_points: "eqd_delta" must be smaller than "step"')
            return -1, 0

        # Store equidistant points
        eqd_rand_points = []

        # Generate equdistant points
        for i in range(0, ceil(chunk_len / eqd_delta)):
            eqd_rand_points.append(
                np.arange(
                    0 + eqd_delta * i,
                    chunk_len + eqd_delta * i,
                    ceil(chunk_len / step),
                )
                % chunk_len
            )

        return eqd_rand_points

    def compute_fingerprint(
        data,
        n_bits,
        power_thresh,
        snr_thresh,
        peak_thresh,
        bias,
        sample_rate,
        eqd_delta,
        ewma_filter=False,
        alpha=0.015,
        remove_noise=False,
        normalize=False,
    ):
        fp = None
        chunk = np.copy(data)
        print("Chunk: ", chunk)

        print("Normalize status: ", normalize)

        if normalize:
            chunk = FastZIP_Protocol.normalize_signal(chunk)

        activity = FastZIP_Protocol.activity_filter(
            chunk, power_thresh, snr_thresh, peak_thresh, sample_rate
        )
        # activity = True
        print("Activity detected:", activity)
        if activity:
            print("Noise removal status: ", remove_noise)
            if remove_noise:
                chunk = FastZIP_Protocol.remove_noise(chunk)
            print("Ewma filter status: ", ewma_filter)
            if ewma_filter:
                chunk = FastZIP_Protocol.ewma_filter(abs(chunk), alpha=alpha)

            qs_thr = FastZIP_Protocol.compute_qs_thr(chunk, bias)
            print("qs threshold", qs_thr)

            pts = FastZIP_Protocol.generate_equidist_points(
                len(data), n_bits, eqd_delta
            )
            print("Points: ", pts)

            fp = ""
            for pt in pts:
                if all(data[pt] > qs_thr):
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
        sample_rate_list,
        eqd_delta_list,
        ewma_filter_list=None,
        alpha_list=None,
        remove_noise_list=None,
        normalize_list=None,
    ):
        def bitstring_to_bytes(s):
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
                normalize = normalize_list[i]

            bits = FastZIP_Protocol.compute_fingerprint(
                data,
                n_bits,
                power_thresh,
                snr_thresh,
                peak_thresh,
                bias,
                sample_rate,
                eqd_delta,
                ewma_filter=ewma_filter,
                alpha=alpha,
                remove_noise=remove_noise,
                normalize=normalize,
            )

            if bits != None:
                key += bits
        return bitstring_to_bytes(key)


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
