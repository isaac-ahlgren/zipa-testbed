import multiprocessing as mp
import os
import math
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple


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