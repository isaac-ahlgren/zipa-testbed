import multiprocessing as mp
import os
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy.cluster import cmeans
from sklearn.decomposition import PCA
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


class IoTCupid_Protocol:
    """
    A protocol designed to process sensor data for event detection, feature extraction,
    and clustering to generate a secure cryptographic key or identifier.

    :param sensors: List of sensor objects used to collect data.
    :param key_length: The length of the cryptographic key to generate.
    :param a: Alpha value used for EWMA filtering.
    :param cluster_sizes_to_check: The range of cluster sizes to evaluate.
    :param feature_dim: The number of principal components to retain in PCA.
    :param quantization_factor: The factor used for quantizing event timings into binary data.
    :param cluster_th: The threshold for determining the elbow in the Fuzzy C-Means clustering.
    :param window_size: The size of the window used for computing derivatives.
    :param bottom_th: Lower threshold for event detection.
    :param top_th: Upper threshold for event detection.
    :param agg_th: Aggregation threshold to decide the significance of an event.
    :param parity_symbols: The number of parity symbols used in error correction.
    :param timeout: Timeout value for protocol operations.
    :param logger: Logger object for recording protocol operations.
    :param verbose: Boolean flag to control verbosity of output.
    """

    def __init__(
        self,
        sensors: List[Any],
        key_length: int,
        a: float,
        cluster_sizes_to_check: int,
        feature_dim: int,
        quantization_factor: int,
        cluster_th: float,
        window_size: int,
        bottom_th: float,
        top_th: float,
        agg_th: int,
        parity_symbols: int,
        timeout: int,
        logger: Any,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the IoTCupid protocol with necessary parameters and configurations.
        """

        self.sensors = sensors

        self.name = "iotcupid"

        self.timeout = timeout

        self.count = 0

        self.verbose = verbose

    def extract_context(self) -> None:
        pass

    def process_context(self) -> Any:
        events = []
        event_features = []
        iteration = 0
        while len(events) < self.min_events:
            chunk = self.read_samples(self.chunk_size)

            smoothed_data = IoTCupid_Protocol.ewma_filter(chunk, self.a)

            derivatives = IoTCupid_Protocol.compute_derivative(smoothed_data, self.window_size)
           
            received_events = IoTCupid_Protocol.detect_events(derivatives, self.bottom_th, self.top_th, self.agg_th)
     
            event_features = IoTCupid_Protocol.get_event_features(recieved_events, signal_data, self.feature_dim)

            # Reconciling lumping adjacent events across windows
            if (
                len(received_events) != 0
                and len(events) != 0
                and received_events[0][0] - events[-1][1] <= self.lump_th
            ):
                events[-1] = (events[-1][0], received_events[0][1])
                length = events[-1][1] - events[-1][0] + 1
                max_amp = np.max([event_features[-1][1], received_event_features[0][1]])
                event_features[-1] = (length, max_amp)

                events.extend(received_events[1:])
                event_features.extend(received_event_features[1:])
            else:
                events.extend(received_events)
                event_features.extend(received_event_features)
            iteration += 1

        # Extracted from read_samples function in protocol_interface
        ProtocolInterface.reset_flag(self.queue_flag)
        self.clear_queue()

        fuzzy_cmeans_w_elbow_method(
            event_features, self.max_clusters, self.cluster_th, self.m_start, self.m_end, self.m_searches
        )

        grouped_events = self.group_events(events, u)

        inter_event_timings = self.calculate_inter_event_timings(grouped_events)

        encoded_timings = self.encode_timings_to_bits(
            inter_event_timings, quantization_factor
        )

        return encoded_timings

    def parameters(self, is_host: bool) -> str:
        pass

    def device_protocol(self, host: Any) -> None:
        pass

    def host_protocol(self, device_sockets: List[Any]) -> None:
        # Log parameters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count))
            print()
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket: Any) -> None:
        pass

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
        for i in range(len(signal) - window_size + 1):
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

    def get_event_features(
        events: List[Tuple[int, int]], sensor_data: np.ndarray, feature_dim: int
    ) -> np.ndarray:
        """
        Extracts features from event data using TSFresh for dimensionality reduction with PCA.

        :param events: List of event indices.
        :param sensor_data: Data from which to extract features.
        :param feature_dim: Dimension of the feature space after PCA.
        :return: Array of reduced dimensionality features.
        """
        timeseries = []
        for i, (start, end) in enumerate(events):
            for time_point in range(start, end):
                timeseries.append((i, time_point, sensor_data[time_point]))

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

        return reduced_dim.to_numpy()

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
        
        best_fpc, best_u, best_cntr = IoTCupid_Protocol.grid_search_cmeans(features, 1, m_start, m_end, m_searches)
        x1 = best_fpc
        rel_val = x1
        c = 1

        prev_fpc = best_fpc
        prev_u = best_u
        prev_cntr = best_cntr
        for i in range(2, max_clusters + 1):

            fpc, u, cntr = IoTCupid_Protocol.grid_search_cmeans(features, i, m_start, m_end, m_searches)
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

    def calculate_cluster_dispersion(
        self, features: np.ndarray, u: np.ndarray, cntr: np.ndarray
    ) -> float:
        """
        Calculates the dispersion of clusters based on membership values and distances to cluster centers.

        :param features: The dataset that has been clustered.
        :param u: The membership matrix from the fuzzy C-means clustering.
        :param cntr: The cluster centers from the fuzzy C-means clustering.
        :return: The dispersion value calculated as the weighted sum of squared distances.
        """
        # Recalculate distances from each sample to each cluster center
        distances = np.zeros(
            (u.shape[0], features.shape[0])
        )  # Initialize distance array
        for j in range(u.shape[0]):  # For each cluster
            for i in range(features.shape[0]):  # For each feature set
                distances[j, i] = np.linalg.norm(features[i] - cntr[j])

        # Calculate dispersion as the weighted sum of squared distances
        dispersion = np.sum(u**2 * distances**2)
        return dispersion

    def grid_search_m(features: np.ndarray, max_clusters: int) -> float:
        """
        Conducts a grid search over possible values of 'm' to find the one that minimizes cluster dispersion.

        :param features: The dataset to be clustered.
        :param max_clusters: The number of clusters to use in the fuzzy C-means algorithm.
        :return: The value of 'm' that resulted in the minimum dispersion.
        """
        best_m = None
        best_score = np.inf

        for m in np.linspace(1.1, 2.0, 10):  # m values from 1.1 to 2.0
            cntr, u, _, _, _, _, _ = cmeans(
                features.T, c=max_clusters, m=m, error=0.005, maxiter=1000
            )
            dispersion = self.calculate_cluster_dispersion(features, u, cntr)
            if dispersion < best_score:
                best_m = m
                best_score = dispersion

        return best_m

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
        grouped_events: List[List[Tuple[int, int]]], Fs, key_size
    ) -> Dict[int, np.ndarray]:
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
                key += in_microseconds.to_bytes(
                    4, "big"
                )  # Going to treat every interval as a 4 byte integer

            if len(key) >= key_size:
                key = bytes(key[:key_size])
                fp.append(key)
        return fp

    def encode_timings_to_bits(
        self, inter_event_timings: Dict[int, np.ndarray], quantization_factor: int = 100
    ) -> Dict[int, str]:
        """
        Encodes the inter-event timings into binary strings by quantizing and converting to binary.

        :param inter_event_timings: The timings between events to encode.
        :param quantization_factor: The factor by which to divide timings for quantization.
        :return: A dictionary with cluster IDs as keys and concatenated binary strings as values.
        """
        encoded_timings = {}
        for cluster_id, timings in inter_event_timings.items():
            quantized_timings = np.floor(timings / quantization_factor).astype(int)
            bit_strings = [format(timing, "b") for timing in quantized_timings]
            encoded_timings[cluster_id] = "".join(bit_strings)
        return encoded_timings

    def extract_column_values(self, df: pd.DataFrame, column_name: str) -> np.ndarray:
        """
        Extracts values from a specified column in a DataFrame.

        :param df: The DataFrame from which values are to be extracted.
        :param column_name: The name of the column from which values are extracted.
        :return: A numpy array containing the values from the specified column.
        """
        return df[column_name].values

    def calculate_cluster_dispersion(
        self, features: np.ndarray, u: np.ndarray, cntr: np.ndarray
    ) -> float:
        """
        Calculates the dispersion of clusters based on membership values and distances to cluster centers.

        :param features: The dataset that has been clustered.
        :param u: The membership matrix from the fuzzy C-means clustering.
        :param cntr: The cluster centers from the fuzzy C-means clustering.
        :return: The dispersion value calculated as the weighted sum of squared distances.
        """
        # Recalculate distances from each sample to each cluster center
        distances = np.zeros(
            (u.shape[0], features.shape[0])
        )  # Initialize distance array
        for j in range(u.shape[0]):  # For each cluster
            for i in range(features.shape[0]):  # For each feature set
                distances[j, i] = np.linalg.norm(features[i] - cntr[j])

        # Calculate dispersion as the weighted sum of squared distances
        dispersion = np.sum(u**2 * distances**2)
        return dispersion

    def iotcupid(
        self,
        raw: pd.DataFrame,
        pre_events: List[Tuple[dt, Any]],
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
    ) -> Tuple[Dict[int, str], List[Tuple[int, int]]]:
        """
        Main function of the IoTCupid Protocol, integrating several preprocessing and analysis steps
        to generate encoded timings from raw sensor data.

        :param raw: DataFrame containing raw sensor data.
        :param pre_events: Pre-processed events data.
        :param key_size: The desired size of the cryptographic key.
        :param Fs: Sampling frequency of the data.
        :param a: Alpha value for EWMA filtering.
        :param cluster_sizes_to_check: Maximum number of clusters to consider.
        :param feature_dim: Number of dimensions for PCA feature reduction.
        :param quantization_factor: Factor for quantizing the inter-event timings.
        :param cluster_th: Threshold to determine the elbow in clustering.
        :param window_size: Size of the window for computing derivatives.
        :param bottom_th: Lower threshold for event detection.
        :param top_th: Upper threshold for event detection.
        :param agg_th: Threshold for aggregating detected events.
        :return: Tuple containing encoded timings and grouped events.
        """
        smoothed_data = IoTCupid_Protocol.ewma_filter(raw, a)

        derivatives = self.compute_derivative(smoothed_data, window_size)

        signal_data = self.extract_column_values(derivatives, "derivative")

        events = self.detect_events(derivatives, bottom_th, top_th, agg_th)
        if len(events) < 2:
            # Needs two events in order to calculate interevent timings
            if self.verbose:
                print("Error: Less than two events detected")
            return ([], events)

        event_features = self.get_event_features(events, signal_data, feature_dim)

        optimal_m = self.grid_search_m(event_features, cluster_sizes_to_check)

        cntr, u, optimal_clusters, fpcs = self.fuzzy_cmeans_w_elbow_method(
            event_features, cluster_sizes_to_check, optimal_m, cluster_th
        )

        grouped_events = self.group_events(events, u)

        inter_event_timings = self.calculate_inter_event_timings(grouped_events)

        encoded_timings = self.encode_timings_to_bits(
            inter_event_timings, quantization_factor
        )

        return encoded_timings, grouped_events