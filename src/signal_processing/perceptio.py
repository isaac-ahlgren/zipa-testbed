from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans


class PerceptioProcessing:
    def perceptio(
        signal: np.ndarray,
        key_size: int,
        Fs: int,
        a: float,
        cluster_sizes_to_check: int,
        cluster_th: float,
        bottom_th: float,
        top_th: float,
        lump_th: int,
    ) -> Tuple[List[bytes], List[Tuple[int, int]]]:
        """
        Processes a signal to extract events, features, and generate fingerprints using k-means clustering.

        :param signal: The input signal to process.
        :param key_size: The size of the key or fingerprint to be generated.
        :param Fs: Sampling frequency of the signal.
        :param a: Smoothing factor used in EWMA for processing the signal.
        :param cluster_sizes_to_check: Maximum number of clusters to evaluate.
        :param cluster_th: Threshold to determine the elbow in k-means clustering.
        :param bottom_th: Lower threshold for event detection.
        :param top_th: Upper threshold for event detection.
        :param lump_th: Threshold for lumping close events together.
        :return: A tuple containing the generated fingerprints and the grouped events.
        """
        events = PerceptioProcessing.get_events(signal, a, bottom_th, top_th, lump_th)

        event_features = PerceptioProcessing.get_event_features(events, signal)

        labels, k, _ = PerceptioProcessing.kmeans_w_elbow_method(
            event_features, cluster_sizes_to_check, cluster_th
        )

        grouped_events = PerceptioProcessing.group_events(events, labels, k)

        fps = PerceptioProcessing.gen_fingerprints(grouped_events, k, key_size, Fs)

        return fps, grouped_events
    
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
            return PerceptioProcessing.ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
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
            return PerceptioProcessing.ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

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
        PerceptioProcessing.ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
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
            PerceptioProcessing.ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                            dtype=dtype, order='C', out=out[-trailing_n:])
        return out

    def ewma(signal: np.ndarray, a: float) -> np.ndarray:
        """
        Computes the exponentially weighted moving average (EWMA) of a signal.

        :param signal: The input signal as a NumPy array.
        :param a: The smoothing factor used in the EWMA calculation.
        :return: The EWMA of the signal as a NumPy array.
        """
        return PerceptioProcessing.ewma_vectorized_safe(signal, a)

    def get_events(
        signal: np.ndarray, a: float, bottom_th: float, top_th: float, lump_th: int
    ) -> List[Tuple[int, int]]:
        """
        Identifies events in a signal based on thresholds and lumping criteria.

        :param signal: The input signal.
        :param a: Smoothing factor for the EWMA.
        :param bottom_th: Lower threshold for event detection.
        :param top_th: Upper threshold for event detection.
        :param lump_th: Threshold for lumping close events together.
        :return: A list of tuples representing the start and end indices of each detected event.
        """
        signal = PerceptioProcessing.ewma(np.abs(signal), a)
 
        in_range = (signal >= bottom_th) & (signal <= top_th)

        in_range = np.insert(in_range, 0, False)

        # Identify the start and end of each event by finding edges in the mask
        diff = np.diff(in_range.astype(int))
        starts = np.where(diff == 1)[0]  # Beginning of events
        ends = np.where(diff == -1)[0]   # End of events

        # Check if the last event goes to the end of the signal
        if in_range[-1]:
            ends = np.append(ends, len(signal))
        
        # Combine starts and ends as tuples in a list
        events = list(zip(starts, ends))

        if len(events) != 0:
            events = PerceptioProcessing.lump_events(events, lump_th)
        return events

    def lump_events(
        events: List[Tuple[int, int]], lump_th: int
    ) -> List[Tuple[int, int]]:
        events = np.array(events)
    
        # Calculate gaps between consecutive events
        gaps = events[1:, 0] - events[:-1, 1]
        
        # Determine where new groups should start based on lump_th
        group_starts = np.where(gaps > lump_th)[0] + 1

        # Create grouping indices for reduceat by inserting a starting index of 0
        indices = np.insert(group_starts, 0, 0)
        
        # Compute minimum starts and maximum ends for each group using reduceat
        merged_starts = np.minimum.reduceat(events[:, 0], indices)
        merged_ends = np.maximum.reduceat(events[:, 1], indices)

        # Stack starts and ends together to form merged events
        merged_events = np.column_stack((merged_starts, merged_ends))

        return merged_events.tolist()

    def generate_features(signal):
        length = len(signal)
        if length == 1:
            max_amplitude = signal[0]
        else:
            max_amplitude = np.max(signal)
        return length, max_amplitude

    def get_event_features(
        events: List[Tuple[int, int]], signal: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Extracts features from each event in a signal.

        :param events: List of tuples with start and end indices for each event.
        :param signal: The original signal from which events were detected.
        :return: A list of tuples containing the length and maximum amplitude of each event.
        """
        event_features = []
        for i in range(len(events)):
            event_signal = signal[events[i][0] : events[i][1]]
            length, max_amplitude = PerceptioProcessing.generate_features(event_signal)
            event_features.append((length, max_amplitude))
        return event_features

    def kmeans_w_elbow_method(
        event_features: List[Tuple[int, float]],
        cluster_sizes_to_check: int,
        cluster_th: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies K-means clustering to event features to determine optimal cluster count using the elbow method.

        :param event_features: List of event features.
        :param cluster_sizes_to_check: Maximum number of clusters to consider.
        :param cluster_th: Threshold for determining the elbow point in clustering.
        :return: Cluster labels and the determined number of clusters.
        """
        if len(event_features) < cluster_sizes_to_check:
            cluster_sizes_to_check = len(event_features)

        km = KMeans(1, n_init=50, random_state=0).fit(event_features)

        x1 = km.inertia_
        rel_inert = x1

        k = 1
        labels = km.labels_
        inertias = [rel_inert]

        for i in range(2, cluster_sizes_to_check + 1):

            labels = km.labels_

            km = KMeans(i, n_init=50, random_state=0).fit(event_features)
            x2 = km.inertia_

            inertias.append(x2)
            perc = (x1 - x2) / rel_inert

            x1 = x2

            # Break if reached elbow
            if perc <= cluster_th:
                k = i - 1
                break

            # Break if reached end
            if i == cluster_sizes_to_check - 1:
                labels = km.labels_
                k = i
                break

        return labels, k, inertias

    def group_events(
        events: List[Tuple[int, int]], labels: np.ndarray, k: int
    ) -> List[List[Tuple[int, int]]]:
        """
        Groups detected events according to their cluster labels.

        :param events: List of detected events.
        :param labels: Cluster labels for each event.
        :param k: Number of clusters.
        :return: A list of lists, where each sublist contains events belonging to the same cluster.
        """
        event_groups = [[] for i in range(k)]
        for i in range(len(events)):
            event_groups[labels[i]].append(events[i])
        return event_groups

    def gen_fingerprints(
        grouped_events: List[List[Tuple[int, int]]], k: int, key_size: int, Fs: int
    ) -> List[bytes]:
        """
        Generates fingerprints from grouped events by calculating the time intervals between them.

        :param grouped_events: List of event groups.
        :param k: Number of clusters.
        :param key_size: Desired key size in bytes.
        :param Fs: Sampling frequency of the original signal.
        :return: List of generated fingerprints.
        """
        from datetime import timedelta

        fp = []
        for i in range(k):
            event_list = grouped_events[i]
            key = bytearray()
            for j in range(len(event_list) - 1):
                interval = (event_list[j + 1][0] - event_list[j][0]) / Fs
                in_microseconds = int(
                    timedelta(seconds=interval) / timedelta(microseconds=1)
                )
                key += in_microseconds.to_bytes(
                    8, "big"
                )  # Going to treat every interval as a 8 byte integer

            if len(key) >= key_size:
                key = bytes(key[-key_size:])
                fp.append(key)
        return fp
