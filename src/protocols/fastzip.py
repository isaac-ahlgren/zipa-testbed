from typing import Any

import numpy as np

from error_correction.fPAKE import fPAKE
from networking.network import ack, ack_standby, send_status, socket
from protocols.protocol_interface import ProtocolInterface
from signal_processing.fastzip import FastZIPProcessing


class FastZIPProtocol(ProtocolInterface):
    def __init__(self, parameters: dict, sensor: Any, logger: Any) -> None:
        super().__init__(parameters, sensor, logger)
        self.name = "Fastzip_Protocol"

        # Store configuration directly
        self.verbose = parameters.get("verbose", False)
        self.timeout = parameters["timeout"]
        self.chunk_size = parameters["chunk_size"]
        self.n_bits = parameters["n_bits"]
        self.power_thresh = parameters["power_thresh"]
        self.snr_thresh = parameters["snr_thresh"]
        self.peak_thresh = parameters["peak_thresh"]
        self.bias = parameters["bias"]
        self.sample_rate = parameters["sample_rate"]
        self.eqd_delta = parameters["eqd_delta"]
        self.peak_status = parameters.get("peak_status", False)
        self.ewma_filter = parameters.get("ewma_filter", False)
        self.alpha = parameters.get("alpha", 0.015)
        self.remove_noise = parameters.get("remove_noise", False)
        self.normalize = parameters.get("normalize", False)
        self.key_length = parameters["key_length"]
        self.parity_symbols = parameters["parity_symbols"]

        # fPAKE related
        self.commitment_length = self.key_length + self.parity_symbols
        self.fPAKE = fPAKE(
            pw_length=self.commitment_length,
            key_length=self.key_length,
            timeout=self.timeout,
        )

    def manage_overlapping_chunks(new_chunk, previous_chunk) -> np.ndarray:
        """
        Generate overlapping chunks from a signal buffer.

        :param new_chunk: The new chunk of signal data.
        :param previous_chunk: The previous chunk of signal data for overlapping.
        :return: A concatenated array of the previous overlap and the new chunk.
        """
        overlap_size = len(new_chunk)
        return np.concatenate((previous_chunk[-overlap_size:], new_chunk))

    def process_context(self) -> Any:
        """
        Processes sensor data to generate cryptographic keys or fingerprints using FastZIP's signal processing capabilities.

        Returns:
            The cryptographic keys or fingerprints generated from the processed data.
        """
        print("Processing context...")
        accumulated_bits = b""
        window_size = self.chunk_size
        overlap_size = window_size // 2

        chunk = self.read_samples(window_size)
        previous_chunk = np.array([])

        while len(accumulated_bits) < self.commitment_length:
            if chunk.size == 0:
                print("No more data to process...")
                break

            processed_bits = self.process_chunk(chunk)
            if processed_bits:
                accumulated_bits += processed_bits
                if len(accumulated_bits) >= self.commitment_length:
                    print(
                        "Reached the required length of bits, stopping further processing."
                    )
                    break

            new_overlap_chunk = self.read_samples(window_size - overlap_size)
            if new_overlap_chunk.size == 0:
                print("No more data available to process.")
                break

            chunk = self.manage_overlapping_chunks(new_overlap_chunk, previous_chunk)
            previous_chunk = chunk

        if len(accumulated_bits) >= self.key_length:
            accumulated_bits = accumulated_bits[: self.commitment_length]

        ProtocolInterface.reset_flag(self.queue_flag)
        self.clear_queue()

        return accumulated_bits

    def process_chunk(self, chunk: np.ndarray) -> bytes:
        """
        Processes a single chunk of data and returns a sequence of bits (as bytes).

        Args:
            chunk (np.ndarray): The data chunk to process.

        Returns:
            bytes: The bits generated from the chunk.
        """
        # Implement the processing as per the FastZIP requirements
        bits = FastZIPProcessing.fastzip_algo(
            sensor_data_list=[chunk],
            n_bits_list=[self.n_bits],
            power_thresh_list=[self.power_thresh],
            snr_thresh_list=[self.snr_thresh],
            peak_thresh_list=[self.peak_thresh],
            bias_list=[self.bias],
            sample_rate_list=[self.sample_rate],
            eqd_delta_list=[self.eqd_delta],
            peak_status_list=[self.peak_status],
            ewma_filter_list=[self.ewma_filter],
            alpha_list=[self.alpha],
            remove_noise_list=[self.remove_noise],
            normalize_list=[self.normalize],
        )
        print(f"Processed bits: {bits}")
        return bits

    def parameters(self, is_host: bool) -> str:
        pass

    def device_protocol(self, host_socket):
        """
        Conducts the device protocol over a given socket.
        """
        host_socket.setblocking(1)
        print("Device: Setting socket to blocking.")
        if self.verbose:
            print("Device: Sending ACK to host...")
        ack(host_socket)

        if not ack_standby(host_socket, self.timeout):
            if self.verbose:
                print("Device: ACK from host not received, exiting...")
            return

        if self.verbose:
            print("Device: ACK received. Running fPAKE protocol...")
        context_data = self.get_context()
        if isinstance(context_data, (bytes, bytearray)):
            context_bytes = context_data
        else:
            context_bytes = bytes(context_data)
        secret_key = self.fPAKE.device_protocol(context_bytes, host_socket)

        if secret_key:
            if self.verbose:
                print("Device: Secret key derived successfully.")
            send_status(host_socket, True)
        else:
            if self.verbose:
                print("Device: Failed to derive the secret key.")
            send_status(host_socket, False)

    def host_protocol_single_threaded(self, device_socket: socket.socket):
        """
        Manages the host-side fPAKE protocol.
        """
        device_socket.setblocking(1)
        print("Host: Setting socket to blocking.")
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("Host: No ACK received within time limit - early exit.")
            return

        if self.verbose:
            print("Host: ACK received from device. Proceeding with fPAKE protocol...")
            print("Host: ACKing participants")
        ack(device_socket)  # Need this to continue protocol; clients were hanging

        fingerprints = self.get_context()
        if isinstance(fingerprints, (bytes, bytearray)):
            fingerprints_bytes = fingerprints
        else:
            fingerprints_bytes = bytes(fingerprints)
        secret_key = self.fPAKE.host_protocol(fingerprints_bytes, device_socket)

        if secret_key:
            if self.verbose:
                print("Host: Key exchange confirmed successfully.")
            send_status(device_socket, True)
        else:
            if self.verbose:
                print("Host: Key exchange confirmation failed.")
            send_status(device_socket, False)
