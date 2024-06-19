#Currently hardcoded for the rms_db column of the audio data
import multiprocessing as mp
import struct

import numpy as np

import chardet

from scipy.signal import chirp, spectrogram
from datetime import datetime as dt

import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from skfuzzy.cluster import cmeans

import matplotlib.pyplot as plt

class IoTCupid_Protocol:
    def __init__(self, verbose=True):
        self.verbose = verbose


    def ewma_filter(self, dataframe, column_name, alpha=0.15):
        #Commented out the normalization
        #dataframe[column_name] = dataframe[column_name] - dataframe[column_name].mean()
        dataframe[column_name] = dataframe[column_name].ewm(alpha=alpha, adjust=False).mean()
        return dataframe



    def compute_derivative(self, signal, window_size):
        signal['timestamp'] = pd.to_datetime(signal['timestamp'])  # Ensure datetime type
        derivative_values = []
        derivative_times = []
        for i in range(window_size, len(signal)):
            window = signal.iloc[i - window_size:i]
            derivative = (window['rms_db'].iloc[-1] - window['rms_db'].iloc[0]) / window_size
            derivative_values.append(derivative)
            derivative_times.append(signal['timestamp'].iloc[i])

        derivative_df = pd.DataFrame({
            'timestamp': derivative_times,
            'derivative': derivative_values
        })
        print("Derivative dataframe:", derivative_df)
        return derivative_df



    def calculate_event_durations(self, events):
        durations = {}
        for event_type, event_list in events.items():
            total_duration = sum((event_list[i+1][0] - event_list[i][0]).total_seconds()
                                for i in range(0, len(event_list)-1, 2))
            average_duration = total_duration / (len(event_list) // 2)
            durations[event_type] = average_duration
        return durations


    def create_event_pairs(self, events):
        all_pairs = {}
        durations = self.calculate_event_durations(events)
        for key, event_list in events.items():
            pairs = []
            margin_noise = durations[key] * 0.5  # Example: Δtn as 50% of the average duration
            margin_signal = durations[key] * 0.25  # Example: Δtv as 25% of the average duration
            for i in range(0, len(event_list), 2):
                if i + 1 < len(event_list):
                    start_event = event_list[i]
                    end_event = event_list[i+1]
                    if start_event[1] in ['on', 1] and end_event[1] in ['off', 0]:
                        pairs.append((start_event[0], end_event[0], margin_noise, margin_signal))
            all_pairs[key] = pairs
        return all_pairs



    def calculate_thresholds(self, all_noise_samples, all_signal_samples):
        if all_noise_samples and all_signal_samples:
            noise_mean, noise_std = np.mean(all_noise_samples), np.std(all_noise_samples)
            signal_mean, signal_std = np.mean(all_signal_samples), np.std(all_signal_samples)
            TL = noise_mean - 2 * noise_std  # Lower threshold
            TU = signal_mean + 2 * signal_std  # Upper threshold
            print(f"Noise Mean: {noise_mean}, Noise Std: {noise_std}, Signal Mean: {signal_mean}, Signal Std: {signal_std}")
            print(f"Calculated TL: {TL}, TU: {TU}")
        else:
            TL, TU = 0, 1  # Default values if no data is available

        return TL, TU


    def collect_noise_signal_samples(self, derivatives, event_pairs):
        all_noise_samples = []
        all_signal_samples = []
        #derivatives are not within the correct timeframes
        for start_timestamp, end_timestamp, margin_noise, margin_signal in event_pairs:
            noise_start = start_timestamp - pd.Timedelta(seconds=margin_noise)
            noise_end = end_timestamp + pd.Timedelta(seconds=margin_noise)
            signal_start = start_timestamp - pd.Timedelta(seconds=margin_signal)
            signal_end = end_timestamp + pd.Timedelta(seconds=margin_signal)

            print("Event Pair Timestamps:")
            print(f"Noise Start to End: {noise_start} to {noise_end}")
            print(f"Signal Start to End: {signal_start} to {signal_end}")

            noise_samples = derivatives[(derivatives['timestamp'] < signal_start) | (derivatives['timestamp'] > signal_end)]['derivative']
            signal_samples = derivatives[(derivatives['timestamp'] >= signal_start) & (derivatives['timestamp'] <= signal_end)]['derivative']
            print(f"Noise Samples Count: {len(noise_samples)}, Signal Samples Count: {len(signal_samples)}")

            all_noise_samples.extend(noise_samples.tolist())
            all_signal_samples.extend(signal_samples.tolist())

        return all_noise_samples, all_signal_samples



    def calculate_global_thresholds(self, raw, events):
        if isinstance(events, dict):
            event_pairs_dict = self.create_event_pairs(events)
        else:
            raise ValueError("Expected a dictionary for events, but received a list.")

        all_noise_samples = []
        all_signal_samples = []

        for event_type, event_pairs in event_pairs_dict.items():
            noise_samples, signal_samples = self.collect_noise_signal_samples(raw, event_pairs)
            all_noise_samples.extend(noise_samples)
            all_signal_samples.extend(signal_samples)

        bottom_th, top_th = self.calculate_thresholds(all_noise_samples, all_signal_samples)
        abs_bottom_th = abs(bottom_th)
        abs_top_th = abs(top_th)

        return abs_bottom_th, abs_top_th



    def evaluate_accuracy(self, detected_events, ground_truth_events, time_threshold=pd.Timedelta(seconds=0.1)):
        true_positives = 0
        matched_gt_events = {}  # Change from set to dictionary

        # Match detected events to ground truth events
        for detected in detected_events:
            detected_start = pd.to_datetime(detected[0])
            for truth in ground_truth_events:
                truth_start = pd.to_datetime(truth[0])
                # Only consider this truth event if it hasn't been matched yet
                if truth not in matched_gt_events and abs(detected_start - truth_start) <= time_threshold:
                    true_positives += 1
                    matched_gt_events[truth] = detected  # Map this truth to the detected event
                    break  # Stop after the first match to prevent multiple detections counting multiple times

        # Precision and recall calculations
        precision = (true_positives / len(detected_events)) * 100 if detected_events else 0
        recall = (true_positives / len(ground_truth_events)) * 100 if ground_truth_events else 0

        # Debug prints
        print(f"True Positives: {true_positives}")
        print(f"Detected Events: {len(detected_events)}")
        print(f"Ground Truth Events: {len(ground_truth_events)}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        return precision, recall


    def grid_search(self, raw_data, events, min_window, max_window, step):
        event_pairs = self.create_event_pairs(events)  # Adjust this call if needed

        best_window_size = min_window
        best_precision = 0
        best_agg_th = 0
        best_TL = 0
        best_TU = 0
        best_derivatives = None

        for agg_th in range(1, 50, 5):
            for window_size in range(min_window, max_window + 1, step):
                derivatives = self.compute_derivative(raw_data, window_size)
                
        
                TL, TU = self.calculate_global_thresholds(derivatives, events)
                print("Lower threshold:", TL)
                print("Upper threshold:", TU)
        
                detected_events = self.detect_events_grid_search(derivatives, TL, TU, agg_th)
                print("Detected events:", detected_events)
       
                flattened_event_pairs = [pair for sublist in event_pairs.values() for pair in sublist]

                precision, recall = self.evaluate_accuracy(detected_events, flattened_event_pairs)
        
                if precision > best_precision:
                    best_precision = precision
                    best_window_size = window_size
                    best_agg_th = agg_th
                    best_TL = TL
                    best_TU = TU
                    best_derivatives = derivatives

                #How should I utilize best recall? 

                print(f"Tested window size: {window_size}, Precision: {precision}, Recall: {recall} TL: {TL}, TU: {TU}, Aggregate Threshold: {agg_th}")

        return best_window_size, best_precision, best_agg_th, best_TL, best_TU, best_derivatives


    def detect_events_grid_search(self, derivatives, bottom_th, top_th, agg_th):
        events = []
        event_start = None
        in_event = False

        timestamps = derivatives['timestamp']  # Extract the timestamp column for mapping indices to timestamps

        # Iterate over the derivative array
        for i in range(len(derivatives)):
            # Check if the absolute value of the derivative is within the thresholds
            if bottom_th <= abs(derivatives['derivative'].iloc[i]) <= top_th:
                if not in_event:
                    event_start = i  # Start of a new event
                    in_event = True
            else:
                if in_event:
                    # End the current event if it exceeds the aggregation threshold
                    if i - event_start > agg_th:
                        # Append the start and end timestamps of the event
                        events.append((timestamps.iloc[event_start], timestamps.iloc[i]))
                    in_event = False

        if in_event:
            events.append((timestamps.iloc[event_start], timestamps.iloc[-1]))

        return events




    def detect_events(self, derivatives, bottom_th, top_th, agg_th):
        events = []
        event_start = None
        in_event = False

        # Iterate over the derivative values directly
        for i, derivative in enumerate(derivatives['derivative']):
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


    def get_event_features(self, events, sensor_data, feature_dim):
        timeseries = []
        for i, (start, end) in enumerate(events):
            for time_point in range(start, end):
                timeseries.append((i, time_point, sensor_data[time_point]))

        df = pd.DataFrame(timeseries, columns=["id", "time", "value"])

        extracted_features = extract_features(df, column_id="id", column_sort="time",
                                              default_fc_parameters=MinimalFCParameters(),
                                              disable_progressbar=True, impute_function=None)

        # Adjust PCA dimensions based on available features
        #replaces features_dim if you want to swap in n_components
        n_features = extracted_features.shape[1]
        print("Number of features extracted:", n_features)
        print("Number of events:", len(events))
        n_components = min(feature_dim, n_features)

        pca = PCA(n_components=feature_dim)
        reduced_dim = pca.fit_transform(extracted_features)

        return reduced_dim


    def fuzzy_cmeans_w_elbow_method(self, features, max_clusters, m, cluster_th):
        # Array to store the Fuzzy Partition Coefficient (FPC)
        fpcs = []
    
        for num_clusters in range(2, max_clusters + 1):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                features, c=num_clusters, m=m, error=0.005, maxiter=1000, init=None, seed=0)
            fpcs.append(fpc)
    
        # Find the elbow point in the FPC array
        optimal_clusters = 2  # Minimum clusters possible
        for i in range(1, len(fpcs) - 1):
            if abs(fpcs[i] - fpcs[i-1]) < cluster_th:
                optimal_clusters = i + 2
                break
    
        # Once optimal clusters are determined, re-run the Fuzzy C-Means with optimal clusters
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            features.T, c=optimal_clusters, m=m, error=0.005, maxiter=1000, init=None)

        return cntr, u, optimal_clusters, fpcs


    def calculate_cluster_dispersion(self, features, u, cntr):
        # Recalculate distances from each sample to each cluster center
        distances = np.zeros((u.shape[0], features.shape[0]))  # Initialize distance array
        for j in range(u.shape[0]):  # For each cluster
            for i in range(features.shape[0]):  # For each feature set
                distances[j, i] = np.linalg.norm(features[i] - cntr[j])

        # Calculate dispersion as the weighted sum of squared distances
        dispersion = np.sum(u**2 * distances**2)
        return dispersion


    def grid_search_m(self, features, max_clusters):
        best_m = None
        best_score = np.inf

        for m in np.linspace(1.1, 2.0, 10):  # m values from 1.1 to 2.0
            cntr, u, _, _, _, _, _ = cmeans(features.T, c=max_clusters, m=m, error=0.005, maxiter=1000)
            dispersion = self.calculate_cluster_dispersion(features, u, cntr)
            if dispersion < best_score:
                best_m = m
                best_score = dispersion

        return best_m


    def group_events(self, events, u):
        # Group events based on maximum membership value
        labels = np.argmax(u, axis=0)
        event_groups = [[] for _ in range(u.shape[0])]
        for label, event in zip(labels, events):
            event_groups[label].append(event)
        return event_groups

    def calculate_inter_event_timings(self, grouped_events):
        inter_event_timings = {}
        for cluster_id, events in enumerate(grouped_events):
            if len(events) > 1:
                start_times = [event[0] for event in events]
                # Calculate time intervals between consecutive events
                intervals = np.diff(start_times)
                inter_event_timings[cluster_id] = intervals
        return inter_event_timings


    def encode_timings_to_bits(self, inter_event_timings, quantization_factor=100):
        encoded_timings = {}
        for cluster_id, timings in inter_event_timings.items():
            quantized_timings = np.floor(timings / quantization_factor).astype(int)
            bit_strings = [format(timing, 'b') for timing in quantized_timings]
            encoded_timings[cluster_id] = ''.join(bit_strings)
        return encoded_timings

    def evaluate_event_detection(self, events):
        # Might need to add more to this method, not much description given for the grid search in the paper
        return len(events)


    def extract_column_values(self, df, column_name):
        return df[column_name].values


    def get_all_event_time_stamps(directory, relative_file_paths, event_name):
        event_dictionary = dict()
        for path, name in zip(relative_file_paths, event_name):
            file_path = directory + path
            events = get_timestamps(file_path)
            event_dictionary[name] = events
        return event_dictionary


    def strings_to_data(events, names_of_events):
        events_considered = []
        for name in names_of_events:
            events_considered.extend(events[name])
        for i in range(len(events_considered)):
            events_considered[i] = dt.strptime(events_considered[i], '%Y-%m-%d %H:%M:%S.%f')


    def iotcupid(
        self,
        raw,
        pre_events,
        key_size,
        Fs,
        a,
        cluster_sizes_to_check,
        feature_dim,
        quantization_factor,
        cluster_th,
    ):
        
        min_window = 1   # Minimum window size in seconds (or appropriate unit)
        max_window = 300 # Maximum window size in seconds (or appropriate unit)
        step = 10        # Step size in seconds (or appropriate unit)

        smoothed_data = self.ewma_filter(raw, 'rms_db', a)
        print("Smoothed data:", smoothed_data)

        window_size, precision, agg_th, TL, TU, derivatives = self.grid_search(smoothed_data, pre_events, min_window, max_window, step)
        print("Best window size:", window_size)
        print("Best detection precision:", precision)
        print("Best aggregate threshold:", agg_th)
        print("Best lower threshold:", TL)
        print("Best upper threshold:", TU)
   
        signal_data = self.extract_column_values(derivatives, 'derivative')

        events = self.detect_events(derivatives, TL, TU, agg_th)
        if len(events) < 2:
            # Needs two events in order to calculate interevent timings
            if self.verbose:
                print("Error: Less than two events detected")
            return ([], events) 

        event_features = self.get_event_features(events, signal_data, feature_dim)

        optimal_m = self.grid_search_m(event_features, cluster_sizes_to_check)

        cntr, u, optimal_clusters, fpcs = self.fuzzy_cmeans_w_elbow_method(event_features, cluster_sizes_to_check, optimal_m, cluster_th) 

        grouped_events = self.group_events(events, u)

        inter_event_timings = self.calculate_inter_event_timings(grouped_events)
        
        encoded_timings = self.encode_timings_to_bits(inter_event_timings, quantization_factor)
        #fps = self.gen_fingerprints(grouped_events, k, key_size, Fs)
        #fps = self.gen_fingerprints(grouped_events, u.shape[0], key_size, Fs)

        return encoded_timings, grouped_events


def load_sensor_datab(filepath):
        df = pd.read_csv(filepath)
        return df


def load_sensor_data(directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    files.sort()

    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        data_frames.append(df)

    full_data = pd.concat(data_frames, ignore_index=True)

    if 'timestamp' in full_data.columns and not pd.api.types.is_datetime64_any_dtype(full_data['timestamp']):
        try:
            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'], errors='coerce')

    full_data.dropna(subset=['timestamp'], inplace=True)

    return full_data


def extract_column_values_raw(df, column_name):
        return df[column_name].values


def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding} (confidence: {result['confidence']})")
        return encoding


def get_timestamps(files_directory):
    import os
    import pandas as pd
    filenames = os.listdir(files_directory)
    events = []
    for f in filenames:
        if f.startswith('.'):  # skip swap files or other non-CSV/temporary files
            continue
        file_path = os.path.join(files_directory, f)
        #encoding = detect_encoding(file_path)
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
            timestamp = dt.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            state = row[column]
            events.append((timestamp, state))
    # Sort events based on timestamp
    events.sort(key=lambda x: x[0])
    return events


def get_all_event_time_stamps(directory, relative_file_paths, event_name):
    event_dictionary = dict()
    for path, name in zip(relative_file_paths, event_name):
        file_path = directory + path
        events = get_timestamps(file_path)
        event_dictionary[name] = events
    return event_dictionary


def strings_to_data(event_dictionary):
    converted_events = {}
    for event_name, timestamps in event_dictionary.items():
        # Convert each timestamp string to a datetime object
        converted_events[event_name] = [dt.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f') for timestamp in timestamps]
    return converted_events

if __name__ == "__main__":
    #fs = 50  # Sampling frequency for bmp280
    fs = 1 # no sampling rate for radiator, coffee_machine, etc.
    directory = os.getcwd() + "/../dataset/"
    relative_file_paths = ["full-coffee/20-coffee", "door-short/20-door"]
    
    event_names = ["COFFEE", "DOOR"]

    events = get_all_event_time_stamps(directory, relative_file_paths, event_names)
    print("Events:", events)


    #directory_path = "/home/isaac/dataset/BMP-backup/10-events"
    directory_path = "/home/isaac/dataset/audio-short/20-audio"

    df = load_sensor_data(directory_path)
    print("Df:", df)

    signal_data = extract_column_values_raw(df, 'rms_db')
    
    print("Signal Data:")
    print(signal_data)
    # Instance of the protocol
    protocol = IoTCupid_Protocol()

    # Define parameters for the iotcupid function
    key_size = 12
    a = 0.15
    cluster_sizes_to_check = 3
    feature_dim = 3
    quantization_factor=100
    cluster_th = 0.08

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
        cluster_th=cluster_th
    )

     # Output results for inspection
    print("Encoded Timings:", encoded_timings)
    print("Grouped Events:", grouped_events)
