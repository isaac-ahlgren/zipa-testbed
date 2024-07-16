# Currently hardcoded for the rms_db column of the audio data
import multiprocessing as mp
import os
import struct
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple, Union

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.signal import chirp, spectrogram
from skfuzzy.cluster import cmeans
from sklearn.decomposition import PCA
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


class IoTCupid_Protocol:
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
        parameters += f"features_dim: {features_dim}\n"
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

    def ewma_filter(self, dataframe: pd.DataFrame, column_name: str, alpha: float = 0.15) -> pd.DataFrame:
        # Commented out the normalization
        # dataframe[column_name] = dataframe[column_name] - dataframe[column_name].mean()
        dataframe[column_name] = (
            dataframe[column_name].ewm(alpha=alpha, adjust=False).mean()
        )
        return dataframe

    def compute_derivative(self, signal: pd.DataFrame, window_size: int) -> pd.DataFrame:
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

    def detect_events(self, derivatives: pd.DataFrame, bottom_th: float, top_th: float, agg_th: int) -> List[Tuple[int, int]]:
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

    def get_event_features(self, events: List[Tuple[int, int], sensor_data: np.ndarray, feature_dim: int) -> np.ndarray:
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
        n_components = min(feature_dim, n_features)

        pca = PCA(n_components=feature_dim)
        reduced_dim = pca.fit_transform(extracted_features)

        return reduced_dim

        def fuzzy_cmeans_w_elbow_method(self, features: np.ndarray, max_clusters: int, m: float, cluster_th: float) -> Tuple[bp.ndarray, np.ndarray, int, List[float]]:
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

    def calculate_cluster_dispersion(self, features: np.ndarray, u: np.ndarray, cntr: np.ndarray) -> float:
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

    def group_events(self, events: List[Tuple[int, int]], u: np.ndarray) -> List[List[Tuple[int, int]]]:
        # Group events based on maximum membership value
        labels = np.argmax(u, axis=0)
        event_groups = [[] for _ in range(u.shape[0])]
        for label, event in zip(labels, events):
            event_groups[label].append(event)
        return event_groups

    def calculate_inter_event_timings(self, grouped_events: List[List[Tuple[int, int]]]) -> Dict[int, np.ndarray]:
        inter_event_timings = {}
        for cluster_id, events in enumerate(grouped_events):
            if len(events) > 1:
                start_times = [event[0] for event in events]
                # Calculate time intervals between consecutive events
                intervals = np.diff(start_times)
                inter_event_timings[cluster_id] = intervals
        return inter_event_timings

    def encode_timings_to_bits(self, inter_event_timings: Dict[int, np.ndarray], quantization_factor: int = 100) -> Dict[int, str]:
        encoded_timings = {}
        for cluster_id, timings in inter_event_timings.items():
            quantized_timings = np.floor(timings / quantization_factor).astype(int)
            bit_strings = [format(timing, "b") for timing in quantized_timings]
            encoded_timings[cluster_id] = "".join(bit_strings)
        return encoded_timings

    def extract_column_values(self, df: pd.DataFrame, column_name: str) -> np.ndarray:
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
        smoothed_data = self.ewma_filter(raw, "rms_db", a)
        print("Smoothed data:", smoothed_data)

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
        # Method does not exist because IoTCupid is different
        # fps = self.gen_fingerprints(grouped_events, k, key_size, Fs)
        # fps = self.gen_fingerprints(grouped_events, u.shape[0], key_size, Fs)

        return encoded_timings, grouped_events


def load_sensor_data(directory: str) -> pd.DataFrame:
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".csv")
    ]

    files.sort()

    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        data_frames.append(df)

    full_data = pd.concat(data_frames, ignore_index=True)

    if "timestamp" in full_data.columns and not pd.api.types.is_datetime64_any_dtype(
        full_data["timestamp"]
    ):
        try:
            full_data["timestamp"] = pd.to_datetime(
                full_data["timestamp"], format="%Y-%m-%d %H:%M:%S.%f"
            )
        except ValueError:
            full_data["timestamp"] = pd.to_datetime(
                full_data["timestamp"], errors="coerce"
            )

    full_data.dropna(subset=["timestamp"], inplace=True)

    return full_data


def extract_column_values_raw(df: pd.DataFrame, column_name: str) -> np.ndarray:
    return df[column_name].values


def detect_encoding(file_path: str) -> Optional[str]: 
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
        encoding = result["encoding"]
        print(f"Detected encoding: {encoding} (confidence: {result['confidence']})")
        return encoding


def get_timestamps(files_directory: str) -> List[Tuple[dt, Any]]: 
    import os

    import pandas as pd

    filenames = os.listdir(files_directory)
    events: List[Tuple[dt, Any]] = []
    for f in filenames:
        if f.startswith("."):  # skip swap files or other non-CSV/temporary files
            continue
        file_path = os.path.join(files_directory, f)
        # encoding = detect_encoding(file_path)
        file = pd.read_csv(file_path)
        if "DOOR" in file.columns:
            column = "DOOR"
        elif "light" in file.columns:
            column = "light"
        elif "thermostat" in file.columns:
            column = "thermostat"
        elif "temperature" in file.columns:
            column = "temperature"
        elif "status" in file.columns:
            column = "status"
        for index, row in file.iterrows():
            timestamp = dt.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            state = row[column]
            events.append((timestamp, state))
    # Sort events based on timestamp
    events.sort(key=lambda x: x[0])
    return events


def get_all_event_time_stamps(directory: str, relative_file_paths: List[str], event_name: List[str]) -> Dict[str, List[Tuple[dt, Any]]]:
    event_dictionary = dict()
    for path, name in zip(relative_file_paths, event_name):
        file_path = directory + path
        events = get_timestamps(file_path)
        event_dictionary[name] = events
    return event_dictionary


# Example Usage:
if __name__ == "__main__":
    fs = 1  # Hz, replace with actual sample rate
    directory = "/home/isaac/dataset/"
    relative_file_paths = ["/full-coffee/20-coffee", "/door-short/20-door"]

    event_names = ["COFFEE", "DOOR"]

    events = get_all_event_time_stamps(directory, relative_file_paths, event_names)
    print("Events:", events)

    # directory_path = "/home/isaac/dataset/BMP-backup/10-events"
    directory_path = "/home/isaac/dataset/audio-short/20-audio"

    df = load_sensor_data(directory_path)
    print("Df:", df)

    signal_data = extract_column_values_raw(df, "rms_db")

    print("Signal Data:")
    print(signal_data)
    # Instance of the protocol
    protocol = IoTCupid_Protocol()

    # Define parameters for the iotcupid function
    key_size = 12
    a = 0.15
    cluster_sizes_to_check = 3
    feature_dim = 3
    quantization_factor = 100
    cluster_th = 0.08
    window_size = 60
    bottom_th = 0.15
    top_th = 0.4
    agg_th = 5

    # Call the iotcupid method
    encoded_timings, grouped_events = protocol.iotcupid(
        raw=df,
        pre_events=events,
        key_size=key_size,
        Fs=fs,
        a=a,
        cluster_sizes_to_check=cluster_sizes_to_check,
        feature_dim=feature_dim,
        quantization_factor=quantization_factor,
        cluster_th=cluster_th,
        window_size=window_size,
        bottom_th=bottom_th,
        top_th=top_th,
        agg_th=agg_th,
    )

    # Output results for inspection
    print("Encoded Timings:", encoded_timings)
    print("Grouped Events:", grouped_events)
