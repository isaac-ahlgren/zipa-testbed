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
    
    def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a vector.
        Will fail for large inputs.
        :param data: Input data
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param offset: optional
            The offset for the moving average, scalar. Defaults to data[0].
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Defaults to 'C'.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the input. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        if data.ndim > 1:
            # flatten input
            data = data.reshape(-1, order)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        if data.size < 1:
            # empty input, return empty array
            return out

        if offset is None:
            offset = data[0]

        alpha = np.array(alpha).astype(dtype, copy=False)

        # scaling_factors -> 0 as len(data) gets large
        # this leads to divide-by-zeros below
        scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                                dtype=dtype)
        # create cumulative sum array
        np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=dtype, out=out)
        np.cumsum(out, dtype=dtype, out=out)

        # cumsums / scaling
        out /= scaling_factors[-2::-1]

        if offset != 0:
            offset = np.array(offset).astype(dtype, copy=False)
            # add offsets
            out += offset * scaling_factors[1:]

        return out
    
    def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a given axis.
        :param data: Input data, must be 1D or 2D array.
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param axis: The axis to apply the moving average on.
            If axis==None, the data is flattened.
        :param offset: optional
            The offset for the moving average. Must be scalar or a
            vector with one element for each row of data. If set to None,
            defaults to the first value of each row.
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Ignored if axis is not None.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the desired output. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)

        assert data.ndim <= 2

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        if data.size < 1:
            # empty input, return empty array
            return out

        if axis is None or data.ndim < 2:
            # use 1D version
            if isinstance(offset, np.ndarray):
                offset = offset[0]
            return FastZIPProcessing.ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                                out=out)

        assert -data.ndim <= axis < data.ndim

        # create reshaped data views
        out_view = out
        if axis < 0:
            axis = data.ndim - int(axis)

        if axis == 0:
            # transpose data views so columns are treated as rows
            data = data.T
            out_view = out_view.T

        if offset is None:
            # use the first element of each row as the offset
            offset = np.copy(data[:, 0])
        elif np.size(offset) == 1:
            offset = np.reshape(offset, (1,))

        alpha = np.array(alpha).astype(dtype, copy=False)

        # calculate the moving average
        row_size = data.shape[1]
        row_n = data.shape[0]
        scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                                dtype=dtype)
        # create a scaled cumulative sum array
        np.multiply(
            data,
            np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                        dtype=dtype)
            / scaling_factors[np.newaxis, :-1],
            dtype=dtype, out=out_view
        )
        np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
        out_view /= scaling_factors[np.newaxis, -2::-1]

        if not (np.size(offset) == 1 and offset == 0):
            offset = offset.astype(dtype, copy=False)
            # add the offsets to the scaled cumulative sums
            out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

        return out

    def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order='C', out=None):
        """
        Reshapes data before calculating EWMA, then iterates once over the rows
        to calculate the offset without precision issues
        :param data: Input data, will be flattened.
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param row_size: int, optional
            The row size to use in the computation. High row sizes need higher precision,
            low values will impact performance. The optimal value depends on the
            platform and the alpha being used. Higher alpha values require lower
            row size. Default depends on dtype.
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Defaults to 'C'.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the desired output. If not provided or `None`,
            a freshly-allocated array is returned.
        :return: The flattened result.
        """
        def get_max_row_size(alpha, dtype=float):
            assert 0. <= alpha < 1.
            # This will return the maximum row size possible on 
            # your platform for the given dtype. I can find no impact on accuracy
            # at this value on my machine.
            # Might not be the optimal value for speed, which is hard to predict
            # due to numpy's optimizations
            # Use np.finfo(dtype).eps if you  are worried about accuracy
            # and want to be extra safe.
            epsilon = np.finfo(dtype).tiny
            # If this produces an OverflowError, make epsilon larger
            return int(np.log(epsilon)/np.log(1-alpha)) + 1
        
        data = np.array(data, copy=False)

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

        if data.size <= row_size:
            # The normal function can handle this input, use that
            return FastZIPProcessing.ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

        if data.ndim > 1:
            # flatten input
            data = np.reshape(data, -1, order=order)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        row_n = int(data.size // row_size)  # the number of rows to use
        trailing_n = int(data.size % row_size)  # the amount of data leftover
        first_offset = data[0]

        if trailing_n > 0:
            # set temporary results to slice view of out parameter
            out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
            data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
        else:
            out_main_view = out
            data_main_view = data

        # get all the scaled cumulative sums with 0 offset
        FastZIPProcessing.ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                        order='C', out=out_main_view)

        scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
        last_scaling_factor = scaling_factors[-1]

        # create offset array
        offsets = np.empty(out_main_view.shape[0], dtype=dtype)
        offsets[0] = first_offset
        # iteratively calculate offset for each row
        for i in range(1, out_main_view.shape[0]):
            offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

        # add the offsets to the result
        out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

        if trailing_n > 0:
            # process trailing data in the 2nd slice of the out parameter
            FastZIPProcessing.ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                            dtype=dtype, order='C', out=out[-trailing_n:])
        return out


    def ewma_filter(data: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply an Exponentially Weighted Moving Average (EWMA) filter to the data.

        :param data: Input signal data.
        :param alpha: Smoothing factor used in the EWMA calculation.
        :return: Filtered signal.
        """
        return FastZIPProcessing.ewma_vectorized_safe(data, alpha)

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
        else:
            peak_thresh = 0

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
