import multiprocessing as mp
import os
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import chardet
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

    def parameters(self, is_host: bool) -> str:
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"alpha: {a}\n"
        parameters += f"cluster_sizes_to_check: {cluster_sizes_to_check}\n"
        parameters += f"features_dim: {self.features_dim}\n"
        parameters += f"quantization_factor: {quantization_factor}\n"
        parameters += f"cluster_th: {cluster_th}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"time_length: {self.time_length}\n"

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

    def ewma_filter(
        self, dataframe: pd.DataFrame, column_name: str, alpha: float = 0.15
    ) -> pd.DataFrame:
        """
        Applies Exponential Weighted Moving Average filtering to a specified column in a DataFrame.

        :param dataframe: Pandas DataFrame containing the data to filter.
        :param column_name: The name of the column to apply the filter on.
        :param alpha: The decay factor for the EWMA filter.
        :return: The DataFrame with the filtered data.
        """
        # Commented out the normalization
        # dataframe[column_name] = dataframe[column_name] - dataframe[column_name].mean()
        dataframe[column_name] = (
            dataframe[column_name].ewm(alpha=alpha, adjust=False).mean()
        )
        return dataframe

    def compute_derivative(
        self, signal: pd.DataFrame, window_size: int
    ) -> pd.DataFrame:
        """
        Computes the derivative of a signal based on a specified window size.

        :param signal: Pandas DataFrame containing the signal data.
        :param window_size: The size of the window over which to compute the derivative.
        :return: DataFrame containing the derivatives.
        """
        signal["timestamp"] = pd.to_datetime(
            signal["timestamp"]
        )  # Ensure datetime type
        derivative_values = []
        derivative_times = []
        for i in range(window_size, len(signal)):
            window = signal.iloc[i - window_size : i]
            derivative = (
                window["rms_db"].iloc[-1] - window["rms_db"].iloc[0]
            ) / window_size
            derivative_values.append(derivative)
            derivative_times.append(signal["timestamp"].iloc[i])

        derivative_df = pd.DataFrame(
            {"timestamp": derivative_times, "derivative": derivative_values}
        )
        print("Derivative dataframe:", derivative_df)
        return derivative_df

    def detect_events(
        self, derivatives: pd.DataFrame, bottom_th: float, top_th: float, agg_th: int
    ) -> List[Tuple[int, int]]:
        """
        Detects events based on derivative thresholds and aggregation criteria.

        :param derivatives: DataFrame containing derivative data.
        :param bottom_th: Lower threshold for derivative to consider an event.
        :param top_th: Upper threshold for derivative to consider an event.
        :param agg_th: Minimum length of an event to be considered significant.
        :return: A list of tuples representing the start and end indices of detected events.
        """
        events = []
        event_start = None
        in_event = False

        # Iterate over the derivative values directly
        for i, derivative in enumerate(derivatives["derivative"]):
            # Check if the absolute value of the derivative is within the thresholds
            if bottom_th <= abs(derivative) <= top_th:
                if not in_event:
                    event_start = i  # Start of a new event
                    in_event = True
            else:
                if in_event:
                    # End the current event if it exceeds the aggregation threshold
                    if i - event_start > agg_th:
                        events.append((event_start, i))
                    in_event = False

        if in_event:
            events.append((event_start, len(derivatives)))

        return events

    def get_event_features(
        self, events: List[Tuple[int, int]], sensor_data: np.ndarray, feature_dim: int
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

        # Adjust PCA dimensions based on available features
        # replaces features_dim if you want to swap in n_components
        n_features = extracted_features.shape[1]
        print("Number of features extracted:", n_features)
        print("Number of events:", len(events))
        # n_components = min(feature_dim, n_features)

        pca = PCA(n_components=feature_dim)
        reduced_dim = pca.fit_transform(extracted_features)

        return reduced_dim

    def fuzzy_cmeans_w_elbow_method(
        self, features: np.ndarray, max_clusters: int, m: float, cluster_th: float
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
        fpcs = []

        for num_clusters in range(2, max_clusters + 1):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                features,
                c=num_clusters,
                m=m,
                error=0.005,
                maxiter=1000,
                init=None,
                seed=0,
            )
            fpcs.append(fpc)

        # Find the elbow point in the FPC array
        optimal_clusters = 2  # Minimum clusters possible
        for i in range(1, len(fpcs) - 1):
            if abs(fpcs[i] - fpcs[i - 1]) < cluster_th:
                optimal_clusters = i + 2
                break

        # Once optimal clusters are determined, re-run the Fuzzy C-Means with optimal clusters
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            features.T, c=optimal_clusters, m=m, error=0.005, maxiter=1000, init=None
        )

        return cntr, u, optimal_clusters, fpcs

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

    def grid_search_m(self, features: np.ndarray, max_clusters: int) -> float:
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
        self, events: List[Tuple[int, int]], u: np.ndarray
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
        self, grouped_events: List[List[Tuple[int, int]]]
    ) -> Dict[int, np.ndarray]:
        """
        Calculates the timings between consecutive events within each group.

        :param grouped_events: The grouped events as determined by the clustering.
        :return: A dictionary with cluster IDs as keys and arrays of inter-event timings as values.
        """
        inter_event_timings = {}
        for cluster_id, events in enumerate(grouped_events):
            if len(events) > 1:
                start_times = [event[0] for event in events]
                # Calculate time intervals between consecutive events
                intervals = np.diff(start_times)
                inter_event_timings[cluster_id] = intervals
        return inter_event_timings

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
        smoothed_data = self.ewma_filter(raw, "rms_db", a)

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