from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy.cluster import cmeans
from sklearn.decomposition import PCA
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

class IoTCupidProcessing:
    def iotcupid(
        signal,
        key_size: int,
        Fs: int,
        a: float,
        cluster_sizes_to_check: int,
        feature_dim: int,
        quantization_factor: int,
        cluster_th: float,
        window_size: int,
        bottom_th: float,
        top_th: float,
        agg_th: int,
        m_start: float,
        m_end: float,
        m_searches: int
    ):
        smoothed_data = IoTCupidProcessing.ewma(signal, a)

        derivatives = IoTCupidProcessing.compute_derivative(smoothed_data, window_size)
           
        received_events = IoTCupidProcessing.detect_events(abs(derivatives), bottom_th, top_th, agg_th)

        received_event_signals = IoTCupidProcessing.get_event_signals(received_events, smoothed_data)
        if len(received_events) < 2:
            # Needs two events in order to calculate interevent timings
            if self.verbose:
                print("Error: Less than two events detected")
            return ([], events)

        event_features = IoTCupidProcessing.get_event_features(received_event_signals, feature_dim)

        cntr, u, optimal_clusters, fpcs  = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
            event_features.T, cluster_sizes_to_check, cluster_th, m_start, m_end, m_searches
        )

        grouped_events = IoTCupidProcessing.group_events(received_events, u)

        inter_event_timings = IoTCupidProcessing.calculate_inter_event_timings(grouped_events, Fs, quantization_factor, key_size)

        return inter_event_timings, grouped_events

    def ewma(signal: np.ndarray, a: float) -> np.ndarray:
        """
        Computes the exponentially weighted moving average (EWMA) of a signal.

        :param signal: The input signal as a NumPy array.
        :param a: The smoothing factor used in the EWMA calculation.
        :return: The EWMA of the signal as a NumPy array.
        """
        y = np.zeros(len(signal))

        y[0] = a * signal[0]
        for i in range(1, len(signal)):
            y[i] = a * signal[i] + (1 - a) * y[i - 1]
        return y

    def compute_derivative(
        signal, window_size: int
    ) -> np.ndarray:
        """
        Computes the derivative of a signal based on a specified window size.

        :param signal: Pandas DataFrame containing the signal data.
        :param window_size: The size of the window over which to compute the derivative.
        :return: DataFrame containing the derivatives.
        """
        derivative_values = []
        for i in range(len(signal) - window_size):
            derivative = (signal[i + window_size] - signal[i]) / window_size
            derivative_values.append(derivative)
        return np.array(derivative_values)

    def detect_events(
        derivatives: np.ndarray, bottom_th: float, top_th: float, agg_th: int
    ) -> List[Tuple[int, int]]:
        """
        Detects events based on derivative thresholds and aggregation criteria.

        :param derivatives: DataFrame containing derivative data.
        :param bottom_th: Lower threshold for derivative to consider an event.
        :param top_th: Upper threshold for derivative to consider an event.
        :param agg_th: Minimum length of an event to be considered significant.
        :return: A list of tuples representing the start and end indices of detected events.
        """
        # Get events that are within the threshold
        events = []
        found_event = False
        beg_event_num = None
        for i in range(len(derivatives)):
            if not found_event and derivatives[i] >= bottom_th and derivatives[i] <= top_th:
                found_event = True
                beg_event_num = i
            elif found_event and (derivatives[i] < bottom_th or derivatives[i] > top_th):
                found_event = False
                found_event = None
                events.append((beg_event_num, i))
        if found_event:
            events.append((beg_event_num, i))

        i = 0
        while i < len(events) - 1:
            if events[i + 1][0] - events[i][1] <= agg_th:
                new_element = (events[i][0], events[i + 1][1])
                events.pop(i)
                events.pop(i)
                events.insert(i, new_element)
            else:
                i += 1

        return events

    def get_event_signals(
        events: List[Tuple[int, int]], sensor_data: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts features from event data using TSFresh for dimensionality reduction with PCA.

        :param events: List of event indices.
        :param sensor_data: Data from which to extract features.
        :param feature_dim: Dimension of the feature space after PCA.
        :return: Array of reduced dimensionality features.
        """
        event_signals = []
        for i, (start, end) in enumerate(events):
            event_signals.append(sensor_data[start : end])

        return event_signals

    def get_event_features(event_signals, feature_dim):
        timeseries = []
        for i in range(len(event_signals)):
            sensor_data = event_signals[i]
            for j in range(len(sensor_data)):
                timeseries.append((i, j , sensor_data[j]))

        df = pd.DataFrame(timeseries, columns=["id", "time", "value"])

        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            impute_function=None,
        )
        pca = PCA(n_components=feature_dim, random_state=0)
        reduced_dim = pca.fit_transform(extracted_features)

        return reduced_dim

    def grid_search_cmeans(features, c, m_start, m_end, m_searches):
        best_cntr = None
        best_u = None
        best_fpc = None
        for m in np.linspace(m_start, m_end, m_searches):  # m values from 1.1 to 2.0
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    features,
                    c=c,
                    m=m,
                    error=0.005,
                    maxiter=100,
                    init=None,
                    seed=0,
                )
            if best_fpc == None or best_fpc < fpc:
                best_fpc = fpc
                best_u = u
                best_cntr = cntr
        return best_fpc, best_u, best_cntr
        

    def fuzzy_cmeans_w_elbow_method(
        features: np.ndarray, max_clusters: int, cluster_th: float, m_start: float, m_end: float, m_searches: int
    ) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
        """
        Performs fuzzy C-means clustering with an elbow method to determine the optimal number of clusters.

        :param features: The dataset to cluster.
        :param max_clusters: The maximum number of clusters to test.
        :param m: The fuzziness parameter for the C-means algorithm.
        :param cluster_th: The threshold for the rate of change in the Fuzzy Partition Coefficient (FPC) to determine the elbow point.
        :return: A tuple containing the cluster centers, the membership matrix, the optimal number of clusters, and the FPC for each number of clusters tested.
        """
        # Array to store the Fuzzy Partition Coefficient (FPC)
        best_fpc, best_u, best_cntr = IoTCupidProcessing.grid_search_cmeans(features, 1, m_start, m_end, m_searches)
        x1 = best_fpc
        rel_val = x1
        c = 1

        prev_fpc = best_fpc
        prev_u = best_u
        prev_cntr = best_cntr
        for i in range(2, max_clusters + 1):

            fpc, u, cntr = IoTCupidProcessing.grid_search_cmeans(features, i, m_start, m_end, m_searches)
            x2 = fpc

            perc = (x1 - x2) / rel_val
            x1 = x2

            # Break if reached elbow
            if perc <= cluster_th or i == max_clusters:
                c = i-1
                best_fpc = prev_fpc
                best_u = prev_u
                best_cntr = prev_cntr
                break

            if i == max_clusters:
                c = i
                best_fpc = fpc
                best_u = u
                best_cntr = cntr

            prev_fpc = fpc
            prev_u = u
            prev_cntr = cntr

        return best_cntr, best_u, c, best_fpc

    def group_events(
        events: List[Tuple[int, int]], u: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """
        Groups detected events based on their highest membership values from fuzzy clustering.

        :param events: The list of events detected.
        :param u: The membership matrix from the fuzzy C-means clustering.
        :return: A list of event groups, each containing events that are grouped together based on clustering.
        """
        # Group events based on maximum membership value
        labels = np.argmax(u, axis=0)
        event_groups = [[] for _ in range(u.shape[0])]
        for label, event in zip(labels, events):
            event_groups[label].append(event)
        return event_groups

    def calculate_inter_event_timings(
        grouped_events: List[List[Tuple[int, int]]], Fs, quantization_factor, key_size
    ):
        """
        Calculates the timings between consecutive events within each group.

        :param grouped_events: The grouped events as determined by the clustering.
        :return: A dictionary with cluster IDs as keys and arrays of inter-event timings as values.
        """
        from datetime import timedelta
        fp = []
        for i in range(len(grouped_events)):
            event_list = grouped_events[i]
            key = bytearray()
            for j in range(len(event_list) - 1):
                interval = (event_list[j + 1][0] - event_list[j][0]) / Fs
                in_microseconds = int(
                    timedelta(seconds=interval) / timedelta(microseconds=1)
                )
                quantized_interval = int(math.floor(in_microseconds / quantization_factor))
                key += in_microseconds.to_bytes(
                    4, "big"
                )  # Going to treat every interval as a 4 byte integer

            if len(key) >= key_size:
                key = bytes(key[:key_size])
                fp.append(key)
        return fp
