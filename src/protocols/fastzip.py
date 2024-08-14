from typing import Any

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from networking.network import (
    ack,
    ack_standby,
    send_status,
    socket,
)

from signal_processing.fastzip import FastZIPProcessing
from error_correction.fPAKE import fPAKE

from protocols.protocol_interface import ProtocolInterface

class FastZIPProtocol(ProtocolInterface):
    def __init__(self, parameters: dict, sensor: Any, logger: Any, network: Any) -> None:
        super().__init__(parameters, sensor, logger)
        self.fPAKE = fPAKE(pw_length=18, key_length=16, timeout=30)  # Example parameters
        self.network = network
        
        self.n_bits = parameters['n_bits']
        self.power_thresh = parameters['power_thresh']
        self.snr_thresh = parameters['snr_thresh']
        self.peak_thresh = parameters['peak_thresh']
        self.bias = parameters['bias']
        self.sample_rate = parameters['sample_rate']
        self.eqd_delta = parameters['eqd_delta']
        self.peak_status = parameters.get('peak_status', False)
        self.ewma_filter = parameters.get('ewma_filter', False)
        self.alpha = parameters.get('alpha', 0.015)
        self.remove_noise = parameters.get('remove_noise', False)
        self.normalize = parameters.get('normalize', False)
    
    def process_context(self) -> Any:
        """
        Processes sensor data to generate cryptographic keys or fingerprints using FastZIP's signal processing capabilities.

        Returns:
            The cryptographic keys or fingerprints generated from the processed data.
        """
        accumulated_bits = b""

        while len(accumulated_bits) < self.key_length:
            chunk = self.read_samples(self.parameters['chunk_size'])
        
            if not chunk.size:
                break

            processed_bits = self.process_chunk(chunk)
            if processed_bits:
                accumulated_bits += processed_bits

        return accumulated_bits[:self.key_length]

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
            n_bits_list=[self.parameters['n_bits']],
            power_thresh_list=[self.parameters['power_thresh']],
            snr_thresh_list=[self.parameters['snr_thresh']],
            peak_thresh_list=[self.parameters['peak_thresh']],
            bias_list=[self.parameters['bias']],
            sample_rate_list=[self.parameters['sample_rate']],
            eqd_delta_list=[self.parameters['eqd_delta']],
            peak_status_list=[self.parameters.get('peak_status', False)],
            ewma_filter_list=[self.parameters.get('ewma_filter', False)],
            alpha_list=[self.parameters.get('alpha', 0.015)],
            remove_noise_list=[self.parameters.get('remove_noise', False)],
            normalize_list=[self.parameters.get('normalize', False)],
        )
        return bits
    

    def parameters(self, is_host: bool) -> str:
        pass


    def device_protocol(self, host_socket):
        """
        Conducts the device protocol over a given socket, handling the protocol's main loop including
        sending ACKs, extracting context, receiving and processing commitments, and performing key confirmation.
        """
        host_socket.setblocking(1)  # Ensure blocking mode for reliable communication

        # Sending ACK to start the communication
        print("Sending ACK to host...")
        ack(host_socket)  # Use network script function to send ACK

        if not ack_standby(host_socket, self.timeout):
            print("ACK from host not received, exiting...")
            return

        print("ACK received. Running fPAKE protocol...")
        context_data = self.get_context()  # Get context data from sensors
        secret_key = self.fPAKE.device_protocol(context_data, host_socket)

        if secret_key:
            print("Secret key derived successfully.")
            hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'fastzip-session')
            session_key = hkdf.derive(secret_key)
            if self.confirm_key_exchange(host_socket, session_key):
                print("Key exchange confirmed successfully.")
                send_status(host_socket, True)  # Communicate success
            else:
                print("Key exchange confirmation failed.")
                send_status(host_socket, False)
        else:
            print("Failed to derive the secret key.")
            send_status(host_socket, False)


    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Manages the host-side fPAKE protocol for secure key exchange using a specific device socket.
        This method handles the host-side logic for establishing a secure connection,
        conducting the fPAKE entropy amplification, and confirming keys with a single client.

        :param device_socket: Socket connected to the client device.
        """
        device_socket.setblocking(1)  # Ensure blocking mode for reliable communication

        # Wait for initial ACK from device to confirm readiness
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK received within time limit - early exit.\n")
            return

        if self.verbose:
            print("ACK received from device. Proceeding with fPAKE protocol...\n")

        # Retrieve context data which are fingerprints for the fPAKE protocol
        fingerprints = self.get_context()

        # Perform the fPAKE protocol using retrieved fingerprints
        try:
            secret_key = self.fPAKE.host_protocol(fingerprints, device_socket)

            if secret_key:
                # Confirm the secret key with the device
                if self.confirm_key_exchange(device_socket, secret_key):
                    if self.verbose:
                        print("Key exchange confirmed successfully.")
                else:
                    if self.verbose:
                        print("Key exchange confirmation failed.")
            else:
                if self.verbose:
                    print("Failed to derive a secret key.")
        except Exception as e:
            if self.verbose:
                print(f"Error during fPAKE protocol execution: {e}")

    def confirm_key_exchange(self, conn, session_key):
        """
        Send and verify the hash of the session key to confirm.

        :param conn: Connection socket.
        :param session_key: The session key derived from the secret.
        :return: True if the key is confirmed, False otherwise.
        """
        try:
            # Send hash of the session key
            key_hash = hashes.Hash(hashes.SHA256())
            key_hash.update(session_key)
            conn.sendall(key_hash.finalize())

            # Receive and verify the response
            partner_hash = conn.recv(1024)
            return partner_hash == key_hash.finalize()
        except Exception as e:
            print(f"Error during key confirmation: {e}")
            return False