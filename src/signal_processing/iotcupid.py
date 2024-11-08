import math
from typing import List, Optional, Tuple

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

        chunks = IoTCupidProcessing.chunk_signal(signal, window_size)

        chunks = IoTCupidProcessing.ewma_on_chunks(chunks, a)

        derivatives = IoTCupidProcessing.compute_derivative_on_chunks(chunks)

        received_events = IoTCupidProcessing.detect_event_on_chunks(
            abs(derivatives), bottom_th, top_th, agg_th, window_size
        )

        received_event_signals = IoTCupidProcessing.get_event_signals(
            received_events, derivatives, window_size
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

    def chunk_signal(signal, window_len):
        output = []
        chunk_num = len(signal) // window_len
        for i in range(chunk_num):
            chunk = signal[i * window_len : (i + 1) * window_len]
            output.append(chunk)
        return output

    def ewma(
        signal_window: np.ndarray, prev_signal_window: np.ndarray, a: float
    ) -> np.ndarray:
        if prev_signal_window is None:
            return signal_window
        else:
            return a * signal_window + (1 - a) * prev_signal_window

    def ewma_on_chunks(chunks, a):
        prev = None
        for i in range(len(chunks)):
            chunks[i] = IoTCupidProcessing.ewma(chunks[i], prev, a)
            prev = chunks[i]
        return chunks

    """
    Comments potentially for the paper: This algorithm doesn't seem like it was designed for live testing in mind.
    This function is a prime example of it. The idea of using a derivative in this algorithm is to try to detect event
    that happen graudally like the temperature changing in the room over half a second when you open up a door. However, in a live system,
    you typically have to process only large chunks at a time. If a gradual event happens between windows, you can't do a sliding window
    derivative between chunks without instense processing overhead (by way of slowly adding new data one at a time and recomputing the derivative each time
    with is extremely computationally expensive for large buffers)
    """

    def compute_derivative(signal):
        return (signal[-1] - signal[0]) / len(signal)

    def compute_derivative_on_chunks(chunks):
        output = np.zeros(len(chunks))
        for i in range(len(chunks)):
            output[i] = IoTCupidProcessing.compute_derivative(chunks[i])
        return output

    def detect_event(derivative, bottom_th, top_th):
        event_detected = False
        if derivative >= bottom_th and derivative <= top_th:
            event_detected = True
        return event_detected

    def detect_event_on_chunks(derivatives, bottom_th, top_th, agg_th, window_size):
        events = None
        for i in range(len(derivatives)):
            derivative = derivatives[i]
            event_detected = IoTCupidProcessing.detect_event(
                derivative, bottom_th, top_th
            )

            if event_detected:
                new_events = [(0, window_size)]
            else:
                new_events = []

            if events is not None:
                events = IoTCupidProcessing.merge_events(
                    events, new_events, agg_th, window_size, i
                )
            else:
                events = new_events
        return events

    def merge_events(
        first_event_list, second_event_list, lump_th, chunk_size, iteration
    ):
        for i in range(len(second_event_list)):
            second_event_list[i] = (
                second_event_list[i][0] + iteration * chunk_size,
                second_event_list[i][1] + iteration * chunk_size,
            )

        event_list = []
        if len(first_event_list) != 0 and len(second_event_list) != 0:
            end_event = first_event_list[-1]
            beg_event = second_event_list[0]

            if beg_event[0] - end_event[1] <= lump_th:
                new_event = (end_event[0], beg_event[1])
                event_list.extend(first_event_list[:-1])
                event_list.append(new_event)
                event_list.extend(second_event_list[1:])
            else:
                event_list.extend(first_event_list)
                event_list.extend(second_event_list)
        else:
            event_list.extend(first_event_list)
            event_list.extend(second_event_list)

        return event_list

    def get_event_signals(
        events: List[Tuple[int, int]],
        sensor_data: np.ndarray,
        window_size,
    ) -> np.ndarray:
        """
        Extract signal segments from sensor data based on provided events.

        :param events: A list of tuples indicating the start and end indices of events in the sensor data.
        :param sensor_data: The complete sensor data array from which events are extracted.
        :return: A list of numpy arrays, each corresponding to the data segment of an event.
        """

        event_signals = []
        for i, (start, end) in enumerate(events):
            deriv_start = start // window_size
            deriv_end = end // window_size
            event_signals.append(sensor_data[deriv_start:deriv_end])

        return event_signals

    def get_event_features(
        event_signals: List[np.ndarray], feature_dim: int
    ) -> np.ndarray:
        """
        Extract features from event signals and reduce dimensions using PCA.

        :param event_signals: List of numpy arrays where each array represents sensor data for an event.
        :param feature_dim: The number of principal components to retain in the PCA.
        :return: A numpy array with reduced dimensions after PCA.
        """
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

    def grid_search_cmeans(
        features: np.ndarray,
        c: int,
        m_start: float,
        m_end: float,
        m_searches: int,
        mem_thresh: float,
    ) -> Tuple[
        Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Perform a grid search to find the best fuzzification parameter for fuzzy c-means clustering.

        :param features: The dataset features to cluster.
        :param c: The number of clusters.
        :param m_start: The start value for the fuzziness parameter `m`.
        :param m_end: The end value for the fuzziness parameter `m`.
        :param m_searches: The number of searches between `m_start` and `m_end`.
        :param mem_thresh: The membership threshold for calculating cluster variance.
        :return: A tuple containing the best score, the best FPC (Fuzzy Partition Coefficient), and the cluster centers and membership functions.
        """
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

        scores = [best_score]

        prev_score = best_score
        prev_fpc = best_fpc
        prev_u = best_u
        prev_cntr = best_cntr
        for i in range(2, max_clusters + 1):

            score, fpc, u, cntr = IoTCupidProcessing.grid_search_cmeans(
                features, i, m_start, m_end, m_searches, mem_thresh
            )
            scores.append(score)
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

        return best_cntr, best_u, c, scores

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
                    8, "big"
                )  # Going to treat every interval as a 8 byte integer

            if len(key) >= key_size:
                key = bytes(key[:key_size])
                fp.append(key)
        return fp

    def calculate_cluster_variance(
        features: np.ndarray, cntr: np.ndarray, u: np.ndarray, mem_thresh: float
    ) -> float:
        """
        Calculate the average variance of clusters based on fuzzy membership and a membership threshold.

        This function calculates the variance by first determining which clusters each feature point is strongly associated with, based on a membership threshold. It then measures the distortion (squared Euclidean distance) of each feature from its associated cluster centers and averages these distortions across all clusters.

        :param features: An array where each column represents a feature point in the space defined by the cluster centers.
        :param cntr: An array where each row represents the coordinates of a cluster center.
        :param u: A matrix of fuzzy membership degrees, where each column corresponds to a feature point and each row corresponds to a cluster.
        :param mem_thresh: A threshold for determining if a feature point is strongly associated with a cluster, based on its membership degree.
        :return: The average distortion as a measure of variance for the clusters.
        """
        labels = IoTCupidProcessing.defuzz(u, mem_thresh)
        # Recalculate distances from each sample to each cluster center
        distortions = np.zeros(cntr.shape[0])
        for i in range(features.shape[0]):
            for j in range(cntr.shape[0]):
                if j in labels[i] or len(labels[i]) == 0:
                    distortions[j] += np.linalg.norm(features[:, i] - cntr[j]) ** 2
        return np.mean(distortions)
