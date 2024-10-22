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

        # Get events that are within the threshold
        events = []
        found_event = False
        beg_event_num = None
        for i in range(len(signal)):
            if not found_event and signal[i] >= bottom_th and signal[i] <= top_th:
                found_event = True
                beg_event_num = i
            elif found_event and (signal[i] < bottom_th or signal[i] > top_th):
                found_event = False
                found_event = None
                events.append((beg_event_num, i))
        if found_event:
            events.append((beg_event_num, i))

        events = PerceptioProcessing.lump_events(events, lump_th)

        return events

    def lump_events(
        events: List[Tuple[int, int]], lump_th: int
    ) -> List[Tuple[int, int]]:
        """
        Combines closely spaced events into single events based on a lumping threshold.

        :param events: A list of tuples, where each tuple contains the start and end indices of an event.
        :param lump_th: The threshold for lumping adjacent events. If the start of the next event minus the end of the current event is less than or equal to this threshold, they will be lumped together.
        :return: A list of tuples with potentially fewer, larger events, where close events have been combined.
        """
        i = 0
        while i < len(events) - 1:
            if events[i + 1][0] - events[i][1] <= lump_th:
                new_element = (events[i][0], events[i + 1][1])
                events.pop(i)
                events.pop(i)
                events.insert(i, new_element)
            else:
                i += 1

        return events

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
            length, max_amplitude = self.generate_features(event_signal)
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
            return np.zeros(len(event_features), dtype=int), 1

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
                    4, "big"
                )  # Going to treat every interval as a 4 byte integer

            if len(key) >= key_size:
                key = bytes(key[-key_size:])
                fp.append(key)
        return fp
