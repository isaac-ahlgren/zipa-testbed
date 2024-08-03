import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
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
        m_searches: int,
        mem_thresh: float,
    ):
        smoothed_data = IoTCupidProcessing.ewma(signal, a)

        derivatives = IoTCupidProcessing.compute_derivative(smoothed_data, window_size)

        received_events = IoTCupidProcessing.detect_events(
            abs(derivatives), bottom_th, top_th, agg_th
        )

        received_event_signals = IoTCupidProcessing.get_event_signals(
            received_events, smoothed_data
        )
        if len(received_events) < 2:
            # Needs two events in order to calculate interevent timings
            print("Error: Less than two events detected")
            return ([], received_events)

        event_features = IoTCupidProcessing.get_event_features(
            received_event_signals, feature_dim
        )

        cntr, u, optimal_clusters, fpcs = (
            IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(
                event_features.T,
                cluster_sizes_to_check,
                cluster_th,
                m_start,
                m_end,
                m_searches,
                mem_thresh,
            )
        )

        grouped_events = IoTCupidProcessing.group_events(received_events, u, mem_thresh)

        inter_event_timings = IoTCupidProcessing.calculate_inter_event_timings(
            grouped_events, Fs, quantization_factor, key_size
        )

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

    """
    Comments potentially for the paper: This algorithm doesn't seem like it was designed for live testing in mind.
    This function is a prime example of it. The idea of using a derivative in this algorithm is to try to detect event
    that happen graudally like the temperature changing in the room over half a second when you open up a door. However, in a live system,
    you typically have to process only large chunks at a time. If a gradual event happens between windows, you can't do a sliding window
    derivative between chunks without instense processing overhead (by way of slowly adding new data one at a time and recomputing the derivative each time
    with is extremely computationally expensive for large buffers)
    """

    def compute_derivative(signal, window_size: int) -> np.ndarray:
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
            if (
                not found_event
                and derivatives[i] >= bottom_th
                and derivatives[i] <= top_th
            ):
                found_event = True
                beg_event_num = i
            elif found_event and (
                derivatives[i] < bottom_th or derivatives[i] > top_th
            ):
                found_event = False
                found_event = None
                events.append((beg_event_num, i - 1))
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
        events: List[Tuple[int, int]],
        sensor_data: np.ndarray,
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
            event_signals.append(sensor_data[start:end])

        return event_signals

    def get_event_features(event_signals, feature_dim):
        timeseries = []
        for i in range(len(event_signals)):
            sensor_data = event_signals[i]
            for j in range(len(sensor_data)):
                timeseries.append((i, j, sensor_data[j]))

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

    def grid_search_cmeans(features, c, m_start, m_end, m_searches, mem_thresh):
        best_cntr = None
        best_u = None
        best_fpc = None
        best_score = None
        for m in np.linspace(m_start, m_end, m_searches):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                features,
                c=c,
                m=m,
                error=0.005,
                maxiter=1000,
                init=None,
                seed=0,
            )
            score = IoTCupidProcessing.calculate_cluster_variance(
                features, cntr, u, mem_thresh
            )
            if best_score is None or score < best_score:
                best_fpc = fpc
                best_u = u
                best_cntr = cntr
                best_score = score
        return best_score, best_fpc, best_u, best_cntr

    """
    There is a couple of issues when using this method that are inherent to elbow method and
    the paper's chosen application of this method. These are just some thoughts to keep in mind for the paper.

    First, the elbow method is fairly unreliable:
    "If one plots the percentage of variance explained by the clusters against the number of clusters,
    the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph.
    The number of clusters is chosen at this point, hence the "elbow criterion".  In most datasets, this "elbow" is ambiguous, making this method subjective and unreliable.
    Because the scale of the axes is arbitrary, the concept of an angle is not well-defined, and even on uniform random data,
    the curve produces an "elbow", making the method rather unreliable." <-- explanation from wikipedia

    Secondly, because we are using the elbow method, I have found no way to make it so it can detect only a single event using synthetic data.
    This is because it is because the elbow method that I have implemented (there is no actual elbow method algorithm) depends on the relative efficency of
    the sequential scores to figure out whether an "elbow" is happening. However, even if there is one cluster, the jump from the using cluster size of 1 to
    2 always makes the score go down significantly making it never pick cluster size 1.

    Thirdly, for both IoTCupid and Perceptio, there references to the elbow method do not come with an algorithm. Particularly, IoTCupid's reference does not
    even contain a mention of the phrase "elbow method" even in the K-means section (https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

    Fourthly, they do not explicitley state how they defuzz their fuzzy cmeans which is necessary to actual assign events into the groups. I
    had to come up with a solution myself which includes a membership threshold argument.

    """

    def fuzzy_cmeans_w_elbow_method(
        features: np.ndarray,
        max_clusters: int,
        cluster_th: float,
        m_start: float,
        m_end: float,
        m_searches: int,
        mem_thresh: int,
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
        best_score, best_fpc, best_u, best_cntr = IoTCupidProcessing.grid_search_cmeans(
            features, 1, m_start, m_end, m_searches, mem_thresh
        )
        x1 = best_score
        rel_val = x1
        c = 1

        prev_score = best_score
        prev_fpc = best_fpc
        prev_u = best_u
        prev_cntr = best_cntr
        for i in range(2, max_clusters + 1):

            score, fpc, u, cntr = IoTCupidProcessing.grid_search_cmeans(
                features, i, m_start, m_end, m_searches, mem_thresh
            )
            x2 = score
            perc = (x1 - x2) / rel_val
            x1 = x2
            # Break if reached elbow
            if perc <= cluster_th or i == max_clusters:
                c = i - 1
                best_fpc = prev_fpc
                best_u = prev_u
                best_cntr = prev_cntr
                best_score = prev_score
                break

            if i == max_clusters:
                c = i
                best_fpc = fpc
                best_u = u
                best_cntr = cntr
                best_score = score

            prev_fpc = fpc
            prev_u = u
            prev_cntr = cntr
            prev_score = score

        return best_cntr, best_u, c, best_score

    def group_events(
        events: List[Tuple[int, int]], u: np.ndarray, mem_thresh
    ) -> List[List[Tuple[int, int]]]:
        """
        Groups detected events based on their highest membership values from fuzzy clustering.

        :param events: The list of events detected.
        :param u: The membership matrix from the fuzzy C-means clustering.
        :return: A list of event groups, each containing events that are grouped together based on clustering.
        """
        labels = IoTCupidProcessing.defuzz(u, mem_thresh)
        event_groups = [[] for _ in range(u.shape[0])]
        for label_list, event in zip(labels, events):
            for label in label_list:
                event_groups[label].append(event)
        return event_groups

    def defuzz(u, mem_thresh):
        labels = []

        for i in range(len(u[0])):
            event_labels = []
            for j in range(len(u[:, i])):
                score = u[j, i]
                if score > mem_thresh:
                    event_labels.append(j)
            labels.append(event_labels)
        return labels

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
                quantized_interval = int(
                    math.floor(in_microseconds / quantization_factor)
                )
                key += quantized_interval.to_bytes(
                    4, "big"
                )  # Going to treat every interval as a 4 byte integer

            if len(key) >= key_size:
                key = bytes(key[:key_size])
                fp.append(key)
        return fp

    def calculate_cluster_variance(features, cntr, u, mem_thresh):
        labels = IoTCupidProcessing.defuzz(u, mem_thresh)
        # Recalculate distances from each sample to each cluster center
        distortions = np.zeros(cntr.shape[0])
        for i in range(features.shape[0]):
            for j in range(cntr.shape[0]):
                if j in labels[i] or len(labels[i]) == 0:
                    distortions[j] += np.linalg.norm(features[:, i] - cntr[j]) ** 2
        return np.mean(distortions)
