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
        self.ACC_FS = 100
        self.GYR_FS = 100
        self.BAR_FS = 10
        self.FP_ACC_CHUNK = 10
        self.FP_GYR_CHUNK = 10
        self.FP_BAR_CHUNK = 20
        self.BITS_ACC = 24
        self.BITS_GYR = 16
        self.BITS_BAR = 12
        self.BIAS_ACC_V = 0.00015
        self.BIAS_ACC_H = 0.0001
        self.BIAS_GYR = 0
        self.BIAS_BAR = 0
        self.DELTA_ACC = 10
        self.DELTA_GYR = 25
        self.DELTA_BAR = 5
        self.FPS_PER_CHUNK = 1000

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

    def get_peaks(self, sig):
        peak_height = np.mean(sorted(sig)[-9:]) * 0.25
        peaks, _ = find_peaks(sig, height=peak_height, distance=0.25*self.sample_rate)
        return len(peaks)

    def activity_filter(self, signal, p_thresh, snr_thresh):
        # Ensure signal is a numpy array
        signal = np.copy(signal)
        
        # Preprocessing common to all sensor types
        # signal = self.ewma_filter(signal, alpha=0.15)  # Adjust alpha as needed
        # signal = gaussian_filter(savgol_filter(signal, 3, 2), sigma=1.4)
        
        # Initialize metrics
        power, snr, n_peaks = 0, 0, 0
        
        # Compute signal power (similar for all sensor types)
        power = self.compute_sig_power(signal)
        
        # Find peaks

        peaks = self.get_peaks(signal)

        # Compute signal's SNR
        snr = self.compute_snr(signal)
        
        # Check against thresholds to determine if activity is present
        activity_detected = False
        if power > p_thresh and snr > snr_thresh:
            activity_detected = True
        
        return activity_detected

    def compute_qs_thr(self, chunk, bias):
        # Make a copy of chunk
        chunk_cpy = np.copy(chunk)
    
        # Sort the chunk
        chunk_cpy.sort()
    
        return np.median(chunk_cpy) + bias
   
    def construct_equidist_rand_points(self, eqd_points1, eqd_points2):
        # Check that length of equidistant arrays is the same
        if len(eqd_points1) != len(eqd_points2):
            print('construct_equidist_rand_points: input arrays must have the same length!')
            return
    
        # Array storing random points
        random_points = np.zeros(len(eqd_points1) * 2, dtype=int)
    
        # Index to populate random points
        idx = 0
    
        # Iterate over equidistant points
        for i in range(0, len(eqd_points1)):
            # Point in the middle is the same for both equ_point arrays
            if i == 0:
                random_points[idx] = eqd_points1[i]
            else:
                # Take one element from the 1st array and the second element from the second array and put the consequently
                random_points[idx + 1] = eqd_points1[i]
                random_points[idx + 2] = eqd_points2[i]
            
                # Move on with idx
                idx += 2
            
        return random_points

    def generate_fingerprint(self, chunk, random_points, qs_thr):
        # Fingerprint to be returned
        fp = []
    
        # Iterate over random points
        for i in range(0, len(random_points)):
            if chunk[random_points[i]] > qs_thr:
                fp.append('1')
            else:
                fp.append('0')
            
        return ''.join(fp)

    def generate_equidist_rand_points(self, chunk_len, step, eqd_delta):
        # Equidistant delta cannot be bigger than the step
        if eqd_delta > step:
            print('generate_equidist_rand_points: "eqd_delta" must be smaller than "step"')
            return -1, 0
    
        # Store equidistant random points
        eqd_rand_points = []
    
        # Generate equdistant points
        for i in range(0, ceil(chunk_len / eqd_delta)):
            eqd_rand_points.append(np.arange(0 + eqd_delta * i, chunk_len + eqd_delta * i, step) % chunk_len)
        
        return eqd_rand_points, len(eqd_rand_points)
    
    def generate_random_points(self, chunk_len, n_bits, strat='random'):
        if strat == 'random':
            return np.array([secretsGenerator.randint(0, chunk_len - 1) for x in range(n_bits)])
        #         return np.array(sorted([secretsGenerator.randint(0, chunk_len - 1) for x in range(n_bits)]))
        elif strat == 'equid-start':
            # Default values for guard and increment
            guard = 0
            inc = 1
        
            # To cover the most of the signal with points we use a guard interval (start, end) and increment (distance between poitns)
            if n_bits == self.BITS_ACC:
                guard = 5
            elif n_bits == self.BITS_GYR:
                guard = 5
                inc = 3
            elif n_bits == self.BITS_BAR:
                guard = 5
                inc = 0
            
            # Cases for reduced number of bits
            if n_bits == self.BITS_ACC / 2 + 1:
                return np.arange(8, 1000, 82)
        
            elif n_bits == self.BITS_GYR / 2 + 1:
                return np.arange(4, 1000, 124)
            
            elif n_bits == self.BITS_BAR / 2 + 1:
                return np.arange(4, 200, 32)
        
            return np.arange(0 + guard, chunk_len + guard, ceil(chunk_len / n_bits) + inc)
        
        elif strat == 'equid-end':
            return np.arange(chunk_len - 1, 0, -ceil(chunk_len / n_bits))
        elif strat == 'equid-midleft' or strat == 'equid-midright':
            # Generate left and right planes with equidistant points
            left_side = np.arange(int(chunk_len / 2), 0, -ceil(chunk_len / n_bits))
            right_side = np.arange(int(chunk_len / 2), chunk_len, ceil(chunk_len / n_bits))
        
            if strat == 'equid-midleft':
                return construct_equidist_rand_points(left_side, right_side)
            else:
                return construct_equidist_rand_points(right_side, left_side)

    def generate_fps_corpus_chunk(self, chunk, chunk_qs_thr, n_bits, n_iter=None, eqd_delta=-1):
        if n_iter is None:
            n_iter = self.FPS_PER_CHUNK
        # Initialize vars for computing a corpus of fingerprints
        fps = []
        rps = []
        eqd_flag = False

        # ToDo: this is just for testing, remove it later on
        rps_strat = ['equid-start', 'equid-end', 'equid-midleft', 'equid-midright']

        # Check if equidistant delta is valid
        if isinstance(eqd_delta, int) and eqd_delta > 0:

            # Use proper increment (distance between poitns) for each modality
            if n_bits == self.BITS_ACC:
                inc = 1
            elif n_bits == self.BITS_GYR:
                inc = 3
            elif n_bits == self.BITS_BAR:
                inc = 0

            # Cases for reduced number of bits
            if n_bits == self.BITS_ACC / 2 + 1:
                inc = 0
                eqd_delta = 50

            elif n_bits == self.BITS_GYR / 2 + 1:
                inc = 0
                eqd_delta = 100

            elif n_bits == self.BITS_BAR / 2 + 1:
                inc = 0
                eqd_delta = 20

            # Generate a corpus of equidistant random points
            eqd_rand_points, n_iter = self.generate_equidist_rand_points(len(chunk), ceil(len(chunk) / n_bits) + inc, eqd_delta)

            # Set equidist flag
            eqd_flag = True

        # Generate a number of fingerprtins from a single chunk
        for i in range(0, n_iter):
            # Generate random x-axis points (time)
            if not eqd_flag:
                rand_points = self.generate_random_points(len(chunk), n_bits, rps_strat[0])
            else:
                rand_points = eqd_rand_points[i]

            # Generate fp
            fp = self.generate_fingerprint(chunk, rand_points, chunk_qs_thr)

            # Store random points
            rps.append(' '.join(str(x) for x in rand_points.tolist()))

            # Append a chunk fingerprint to the corpus of fingerprints
            fps.append(''.join(fp))

        return fps, rps

    def quantize_signal(self, signal):
        # Quantization scheme to convert the filtered signal into binary data
        threshold = np.mean(signal)  # Using mean as a simple threshold example
        bits = ''.join(['1' if s > threshold else '0' for s in signal])
        return bits

    def signal_to_bits(self, signal, sensor_type):
        # Determine chunk and window sizes based on sensor_type
        if sensor_type == 'bar':
            fp_chunk = self.FP_BAR_CHUNK
            fs = self.BAR_FS
            n_bits = self.BITS_BAR
            bias = self.BIAS_BAR
        elif sensor_type == 'acc':
            fp_chunk = self.FP_ACC_CHUNK
            fs = self.ACC_FS
            n_bits = self.BITS_ACC
            bias = self.BIAS_ACC_V
        elif sensor_type == 'gyrW':
            fp_chunk = self.FP_GYR_CHUNK
            fs = self.GYR_FS
            n_bits = self.BITS_GYR
            bias = self.BIAS_GYR
        # Define for other sensor types as necessary

        # Determine the window size for chunking
        win_size = int(fp_chunk / 4) if sensor_type == 'bar' else int(fp_chunk / 2)

        # Compute number of chunks
        n_chunks = int(len(signal) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)

        binary_data = []
        for i in range(n_chunks):
            # Get signal chunk
            sig_chunk = signal[i * win_size * fs:(i * win_size + fp_chunk) * fs]

            # Process each chunk if it contains significant activity
            if sensor_type == 'bar':
                activity_detected = self.activity_filter(self.normalize_signal(sig_chunk, 'meansub'), sensor_type)
            else:
                activity_detected = self.activity_filter(sig_chunk, sensor_type)

            if activity_detected:
                processed_chunk = self.process_chunk(sig_chunk, sensor_type)
                chunk_qs_thr = self.compute_qs_thr(processed_chunk, bias)
                fps, rps = self.generate_fps_corpus_chunk(processed_chunk, chunk_qs_thr, n_bits, 1)
                binary_data.append({'fps': fps, 'rps': rps})
            else:
                binary_data.append('No significant activity detected')

        return binary_data

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


