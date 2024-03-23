import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from math import ceil

class FastZIP_Protocol:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate  # Sensor's sampling rate
        self.G_UNIT = 9.80665  # Earth's gravity in m/s^2
        self.P_THR_ACC_V = -30.4
        self.P_THR_ACC_H = -28
        self.P_THR_GYR = -36
        self.P_THR_BAR = -12
        self.SNR_THR_ACC_V = 1.08
        self.SNR_THR_ACC_H = 1.32
        self.SNR_THR_GYR = 0.82
        self.SNR_THR_BAR = 1.2
        self.NP_THR_ACC_V = 11
        self.NP_THR_ACC_H = 14
        self.EWMA_ACC_V = 0.16
        self.EWMA_ACC_H = 0.2
        self.RNG_THR_BAR = 1.08
        self.NORM_TYPES = ['energy', 'zscore', 'meansub', 'minmax']
        self.PEAK_HEIGHT = 0.25
        self.PEAK_DISTANCE = 0.25 * self.sample_rate

    def align_adv_data(self, signal):
        # Assuming alignment is based on pre-calculated offsets or experimental adjustments
        # Placeholder for actual alignment logic
        aligned_signal = signal
        return aligned_signal

    def normalize_signal(self, sig, norm):
        # Check if sig is non-zero
        if len(sig) == 0:
            print('normalize_signal: signal must have non-zero length!')
            return

        if not isinstance(norm, str) or norm not in self.NORM_TYPES:
            print('normalize_signal: provide one of the supported normalization methods %s as a string paramter!' % self.NORM_TYPES)
            return

        # Noramlized signal to be returned
        norm_sig = np.copy(sig)

        # Check how signal should be normalized
        if norm == 'energy':
            # Perform energy normalization
            norm_sig = norm_sig / np.sqrt(np.sum(norm_sig ** 2))

        elif norm == 'zscore':
            # Perform z-score normalization (also knonw as variance scaling)
            if np.std(norm_sig) != 0:
                norm_sig = (norm_sig - np.mean(norm_sig)) / np.std(norm_sig)
            else:
                print('normalize_signal: cannot perform Z-score normalization, STD is zero --> returning the original signal!')
                return sig

        elif norm == 'meansub':
            # Subtract mean from sig
            norm_sig = norm_sig - np.mean(norm_sig)

        elif norm == 'minmax':
            # Perform min-max normalization
            if np.amax(norm_sig) - np.amin(norm_sig) != 0:
                norm_sig = (norm_sig - np.amin(norm_sig)) / (np.amax(norm_sig) - np.amin(norm_sig))
            else:
                print('normalize_signal: cannot perform min-max normalization, min == max --> returning the original signal!')
                return sig

        return norm_sig

    def remove_noise(self, data, apply_to):
        # Array to store filtered acc data
        rn_data = np.zeros(data.shape)
    
        if apply_to == 'all':
            # Filter depending on dimensions
            if len(data.shape) == 1:
                # Apply first Savitzky–Golay and then Gaussian filters to data
                rn_data = gaussian_filter(savgol_filter(data, 3, 2), sigma=1.4)
            
            elif len(data.shape) > 1:
                for i in range(0, data.shape[1]):
                    # Apply first Savitzky–Golay and then Gaussian filters to each column
                    rn_data[:, i] = gaussian_filter(savgol_filter(data[:, i], 3, 2), sigma=1.4)
            else:
                print('remove_noise: data must have a non-zero dimension!')
                sys.exit(0)
        
        elif apply_to == 'chunk':
            # Here we only deal with 1D data, no need to check the dimensions
            rn_data = gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)
        
        else:
            print('remove_noise: "apply_to" parameter can only be "all" or "chunk"!')
            sys.exit(0)
        
        return rn_data


    def process_chunk(self, signal, sensor_type):
        """
        Process the signal chunk based on sensor type if it passes the AR criteria.
        """
        if sensor_type == 'bar':
            # Normalize signal for barometric data
            signal = self.normalize_signal(signal, norm='meansub')
        elif sensor_type in ['acc_v', 'acc_h', 'gyrW']:
            # For accelerometer and gyroscope, remove noise as necessary
            signal = self.remove_noise(signal, apply_to='all')
            if sensor_type in ['acc_v', 'acc_h']:
                # Apply EWMA filter with specific alpha based on sensor orientation
                alpha = self.EWMA_ACC_V if sensor_type == 'acc_v' else self.EWMA_ACC_H
                signal = self.ewma_filter(np.abs(signal), alpha=alpha)
        return signal

    def ewma_filter(self, data, alpha=0.15):
        ewma_data = np.zeros(len(data))
        ewma_data[0] = data[0]
        for i in range(1, len(ewma_data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        return ewma_data

    def compute_sig_power(self, sig):
        if len(sig) == 0:
            print('compute_sig_power: signal must have non-zero length!')
            return
        return 10 * np.log10(np.sum(sig ** 2) / len(sig))

    def compute_snr(self, sig):
        if len(sig) == 0:
            print('compute_snr: signal must have non-zero length!')
            return
        return np.mean(abs(sig)) / np.std(abs(sig))

    def get_acc_peaks(self, sig):
        peak_height = np.mean(sorted(sig)[-9:]) * self.PEAK_HEIGHT
        peaks, _ = find_peaks(sig, height=peak_height, distance=self.PEAK_DISTANCE)
        return len(peaks)

    def activity_filter(self, signal, sensor_type):
        # Ensure signal is a numpy array
        signal = np.copy(signal)
        
        # Preprocessing common to all sensor types
        # signal = self.ewma_filter(signal, alpha=0.15)  # Adjust alpha as needed
        # signal = gaussian_filter(savgol_filter(signal, 3, 2), sigma=1.4)
        
        # Initialize metrics
        power, snr, n_peaks = 0, 0, 0
        
        # Compute signal power (similar for all sensor types)
        power = self.compute_sig_power(signal)
        
        # For accelerometer data: additional processing for n_peaks and snr estimation
        if sensor_type in ['acc_v', 'acc_h']:
            alpha = self.EWMA_ACC_V if sensor_type == 'acc_v' else self.EWMA_ACC_H
            # Smooth signal further with EWMA filter
            signal = self.ewma_filter(np.abs(signal), alpha)
            
            # Compute number of prominent peaks
            _, peaks = self.get_acc_peaks(signal, self.sample_rate)
            n_peaks = len(peaks)
        
        # Compute signal's SNR (similar for all sensor types)
        snr = self.compute_snr(signal)
        
        # Check against thresholds to determine if activity is present
        activity_detected = False
        if sensor_type == 'acc_v' and power > self.P_THR_ACC_V and snr > self.SNR_THR_ACC_V and n_peaks > self.NP_THR_ACC_V:
            activity_detected = True
        elif sensor_type == 'acc_h' and power > self.P_THR_ACC_H and snr > self.SNR_THR_ACC_H and n_peaks > self.NP_THR_ACC_H:
            activity_detected = True
        elif sensor_type == 'gyrW' and power > self.P_THR_GYR and snr > self.SNR_THR_GYR:
            activity_detected = True
        elif sensor_type == 'bar' and power > self.P_THR_BAR and snr > self.SNR_THR_BAR:
            activity_detected = True
        
        return activity_detected

    def decompose_acc(self, data, sw_len=5, sw_step=1, convert_flag=True):
        # Convert sliding window length and step from seconds to # of samples
        sw_len = sw_len * self.sample_rate
        sw_step = sw_step * self.sample_rate

            # Array of resulting vertical and horizontal acc components
        dec_acc = np.zeros((len(data), 2))

        for i in range(0, ceil(len(data) / sw_len)):
            # Get a submatrix of length sw_len
            sw_frame = data[i * sw_len:(i + 1) * sw_len, :].copy()

            if sw_frame.shape[1] != 3:
                raise ValueError('decompose_acc: provided "data" should be three dimensional!')

            # Gravity estimation over each column X, Y and Z
            g_est = np.mean(sw_frame, axis=0)

            # Remove gravity from sw_frame
            sw_frame -= g_est

            # Convert acc values from m/s^2 to G units if flag is True
            if convert_flag:
                sw_frame /= self.G_UNIT
                g_est /= self.G_UNIT

            # Compute the vertical component as a dot product
            v_acc = np.dot(sw_frame, g_est) / np.linalg.norm(g_est)
            # Compute the horizontal component
            h_acc = np.linalg.norm(sw_frame - np.outer(v_acc, g_est) / np.linalg.norm(g_est)**2, axis=1)

            # Update dec_acc
            dec_acc[i * sw_len:(i + 1) * sw_len, 0] = v_acc
            dec_acc[i * sw_len:(i + 1) * sw_len, 1] = h_acc

        return dec_acc

    def quantize_signal(self, signal):
        # Quantization scheme to convert the filtered signal into binary data
        threshold = np.mean(signal)  # Using mean as a simple threshold example
        bits = ''.join(['1' if s > threshold else '0' for s in signal])
        return bits

    def signal_to_bits(self, signal, sensor_type='acc'):
        if sensor_type == 'bar':
            activity_detected = self.activity_filter(self.normalize_signal(signal, 'meansub'), sensor_type);
        else:
            activity_detected = self.activity_filter(signal, sensor_type);

        if activity_detected:
            processed_signal = self.process_chunk(signal, sensor_type)
            bits = self.quantize_signal(processed_signal)
            return bits
        else:
            return 'No significant activity detected'

# Example usage:
if __name__ == "__main__":
    sample_rate = 100  # Hz, replace with actual sample rate
    
    # Simulate 1D barometer data (pressure readings)
    bar_signal = np.random.normal(1013, 5, size=sample_rate * 60)  # 1 minute of data
    # Example accelerometer data; replace with actual sensor data
    acc_signal = np.random.normal(0, 1, size=(sample_rate * 60, 3))  # Assuming 3-axis accelerometer data for 1 minute
    # Simulate 3D gyroscope data (angular velocity)
    gyr_signal = np.random.normal(0, 1, size=(sample_rate * 60, 3))  # 1 minute of 3-axis data

    fastzip = FastZIP_Protocol(sample_rate)

    bar_binary_data = fastzip.signal_to_bits(bar_signal, sensor_type='bar')
    print("Barometer binary data:", bar_binary_data)
    acc_binary_data = fastzip.signal_to_bits(acc_signal, sensor_type='acc')
    print("Accelerometer binary data:", acc_binary_data)
    gyr_binary_data = fastzip.signal_to_bits(gyr_signal, sensor_type='gyrW')
    print("Gyroscope binary data:", gyr_binary_data)


