# TODO compare with Seemoo lab implementation: https://github.com/seemoo-lab/ubicomp19_zero_interaction_security/blob/master/Visualization/Miettinen.ipynb
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj
from networking.network import dh_exchange, send_nonce_msg, socket


class Miettinen_Protocol:
    def __init__(
        self,
        sensor: Any,
        key_length: int,
        parity_symbols: int,
        f: float,
        w: float,
        rel_thresh: float,
        abs_thresh: float,
        auth_threshold: float,
        success_threshold: int,
        max_iterations: int,
        timeout: int,
        logger: Any,
        verbose: bool = True,
    ):
        """
        Initializes a new instance of the Miettinen Protocol with specified parameters for key generation and communication.

        :param sensor: The sensor object that provides access to real-time data.
        :param key_length: The length of the cryptographic key to be generated.
        :param parity_symbols: The number of parity symbols used in the error-correction scheme.
        :param f: The frame rate factor, used to calculate the window size for the signal processing.
        :param w: The window size factor, used alongside the frame rate to define the granularity of signal analysis.
        :param rel_thresh: The relative threshold for feature extraction in key generation.
        :param abs_thresh: The absolute threshold for feature extraction in key generation.
        :param auth_threshold: The threshold for successful authentication attempts before considering the session authenticated.
        :param success_threshold: The number of successful key exchanges required for the protocol to be considered successful.
        :param max_iterations: The maximum number of iterations (key exchanges) the protocol should attempt.
        :param timeout: The timeout in seconds for network operations.
        :param logger: A logging object used to record protocol activity and debugging information.
        :param verbose: A boolean flag that indicates whether to output detailed debug information.
        """
        self.sensor = sensor
        self.f = int(f * self.sensor.sensor.sample_rate)
        self.w = int(w * self.sensor.sensor.sample_rate)
        self.rel_thresh = rel_thresh
        self.abs_thresh = abs_thresh
        self.auth_threshold = auth_threshold
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations

        self.timeout = timeout
        self.name = "miettinen"

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, key_length), key_length
        )
        self.hash_func = hashes.SHA256()
        self.ec_curve = ec.SECP384R1()
        self.nonce_byte_size = 16

        self.time_length = (w + f) * (self.commitment_length * 8 + 1)

        self.logger = logger

        self.count = 0

        self.verbose = verbose

    def signal_preprocessing(
        self, signal: np.ndarray, no_snap_shot_width: int, snap_shot_width: int
    ) -> np.ndarray:
        """
        Processes the given signal into chunks based on specified snapshot widths and calculates the average of each chunk.

        :param signal: The raw signal data as a numpy array.
        :param no_snap_shot_width: Width of non-snapshot portion of the signal in samples.
        :param snap_shot_width: Width of the snapshot portion of the signal in samples.
        :return: A numpy array containing the mean value of each snapshot segment.
        """
        block_num = int(len(signal) / (no_snap_shot_width + snap_shot_width))
        c = np.zeros(block_num)
        for i in range(block_num):
            c[i] = np.mean(
                signal[
                    i
                    * (no_snap_shot_width + snap_shot_width) : i
                    * (no_snap_shot_width + snap_shot_width)
                    + snap_shot_width
                ]
            )
        return c

    def gen_key(self, c: np.ndarray, rel_thresh: float, abs_thresh: float) -> str:
        """
        Generates a key based on the relative and absolute thresholds applied to the processed signal.

        :param c: The processed signal data from `signal_preprocessing`.
        :param rel_thresh: The relative threshold for generating bits.
        :param abs_thresh: The absolute threshold for generating bits.
        :return: A binary string representing the generated key.
        """
        bits = ""
        for i in range(len(c) - 1):
            feature1 = np.abs((c[i] / c[i - 1]) - 1)
            feature2 = np.abs(c[i] - c[i - 1])
            if feature1 > rel_thresh and feature2 > abs_thresh:
                bits += "1"
            else:
                bits += "0"
        return bits

    def miettinen_algo(self, x: np.ndarray) -> bytes:
        """
        Main algorithm for key generation using signal processing and threshold-based key derivation.

        :param x: The raw signal data.
        :param f: The frame rate factor, used to calculate the window size for the signal processing.
        :param w: The window size factor, used alongside the frame rate to define the granularity of signal analysis.
        :param rel_thresh: The relative threshold for feature extraction in key generation.
        :param abs_thresh: The absolute threshold for feature extraction in key generation.
        :return: A byte string of the generated key.
        """

        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        signal = Miettinen_Protocol.signal_preprocessing(x, f, w)
        key = Miettinen_Protocol.gen_key(signal, rel_thresh, abs_thresh)
        return bitstring_to_bytes(key)

    def extract_context(self) -> Tuple[bytes, np.ndarray]:
        """
        Reads the sensor data and applies the Miettinen algorithm to extract cryptographic key material.

        :return: A tuple of the generated key bits and the raw sensor signal.
        """
        signal = self.sensor.read(
            int(self.time_length * self.sensor.sensor.sample_rate)
        )
        bits = Miettinen_Protocol.miettinen_algo(
            signal, self.f, self.w, self.rel_thresh, self.abs_thresh
        )
        return bits, signal

    def parameters(self, is_host: bool) -> str:
        """
        Constructs a string detailing the current settings and parameters of the protocol.

        :param is_host: Boolean indicating whether the device is host.
        :return: A string containing formatted protocol parameters.
        """
        parameters = ""
        parameters += "protocol: " + self.name
        parameters += "is_host: " + str(is_host)
        parameters += (
            "\nsensor: " + self.sensor.sensor.name
        )  # Name's in physical sensor.
        parameters += "\nkey_length: " + str(self.key_length)
        parameters += "\nparity_symbols: " + str(self.parity_symbols)
        parameters += "\nf: " + str(self.f)
        parameters += "\nw: " + str(self.w)
        parameters += "\nrel_thresh: " + str(self.rel_thresh)
        parameters += "\nabs_thresh: " + str(self.abs_thresh)
        parameters += "\nsuccess_threshold: " + str(self.success_threshold)
        parameters += "\nauthentication_threshold: " + str(self.auth_threshold)
        parameters += "\nmax_iterations: " + str(self.max_iterations)
        parameters += "\ntime_length: " + str(self.time_length)
        return parameters

    def device_protocol(self, host_socket: socket.socket) -> None:
        """
        Implements the device side of the protocol, handling communications and key exchange operations.

        :param host_socket: The socket connected to the host.
        """
        host_socket.setblocking(1)

        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("\nSending ACK")
        ack(host_socket)

        # Wait for ack from host to being context extract, quit early if no response within time
        if self.verbose:
            print("\nWaiting for ACK from host")

        if not ack_standby(host_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit\n")
            return

        # Shared key generated
        shared_key = self.diffie_hellman(host_socket)

        current_key = shared_key
        successes = 0
        total_iterations = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            # Sending ack that they are ready to begin

            if self.verbose:
                print("Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Sending ACK\n")
            ack(host_socket)

            success = False

            # Extract bits from sensor
            witness, signal = self.extract_context()

            # Wait for Commitment
            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)

            commitment = commitments[0]

            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return

            if self.verbose:
                print("witness: " + str(witness))

            # Decommit

            if self.verbose:
                print("Decommiting")
            prederived_key = self.re.decommit_witness(commitment, witness)

            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Send nonce message to host
            generated_nonce = self.send_nonce_msg_to_host(
                host_socket, pd_key_hash, derived_key
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                return

            # If hashes are equal, then it was successful
            if self.verify_mac_from_host(
                recieved_nonce_msg, generated_nonce, derived_key
            ):
                success = True
                successes += 1
                current_key = derived_key

            if self.verbose:
                print("Produced Key: " + str(derived_key))
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(total_iterations)
                )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=total_iterations,
            )

            # Increment total number of iterations key evolution has occured
            total_iterations += 1

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / total_iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / total_iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(total_iterations)
                    + " succeeded: "
                    + str(successes / total_iterations >= self.auth_threshold),
                )
            ]
        )

        self.count += 1

    def host_protocol(self, device_sockets: List[socket.socket]) -> None:
        """
        Manages the host side of the protocol, coordinating with multiple devices.

        :param device_sockets: A list of sockets, each connected to a device participating in the protocol.
        """

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Handles protocol operations for a single device from the host's perspective in a separate thread.

        :param device_socket: The socket for communication with a specific device.
        """
        device_socket.setblocking(1)

        device_ip_addr, device_port = device_socket.getpeername()

        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return

        if self.verbose:
            print("ACKing participating device")
        ack(device_socket)

        # Shared key generated
        shared_key = self.diffie_hellman(device_socket)

        current_key = shared_key
        total_iterations = 0
        successes = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            success = False
            # ACK device
            if self.verbose:
                ack(device_socket)

            if not ack_standby(device_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Successfully ACKed participating device\n")

            # Extract key from sensor
            witness, signal = self.extract_context()

            # Commit Secret
            if self.verbose:
                print("Commiting Witness")
            prederived_key, commitment = self.re.commit_witness(witness)

            if self.verbose:
                print("witness: " + str(witness))
                print()

            if self.verbose:
                print("Sending commitment")
                print()

            send_commit([commitment], None, device_socket)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                    print()
                return

            # Derive new key using previous key and new prederived key from fuzzy commitment
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            if self.verify_mac_from_device(
                recieved_nonce_msg, derived_key, pd_key_hash
            ):
                success = True
                successes += 1
                current_key = derived_key

            # Create and send key confirmation value
            self.send_nonce_msg_to_device(
                device_socket, recieved_nonce_msg, derived_key, pd_key_hash
            )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=total_iterations,
                ip_addr=device_ip_addr,
            )

            # Increment total times key evolution has occured
            total_iterations += 1

            if self.verbose:
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(total_iterations)
                )
                print()

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / total_iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / total_iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(total_iterations)
                    + " succeeded: "
                    + str(successes / total_iterations >= self.auth_threshold),
                )
            ],
            ip_addr=device_ip_addr,
        )

        self.count += 1

    # TODO: Refactor by putting in common_protocols.py, then change all references to this version to the common_protocols.py version
    def diffie_hellman(self, socket: socket.socket) -> bytes:
        """
        Performs the Diffie-Hellman key exchange over a given socket to securely generate a shared secret.

        :param socket: The socket object used for the exchange.
        :return: The derived shared key as a byte string.
        """
        # Generate initial private key for Diffie-Helman
        initial_private_key = ec.generate_private_key(self.ec_curve)

        public_key = initial_private_key.public_key().public_bytes(
            Encoding.X962, PublicFormat.CompressedPoint
        )

        # Send initial key for Diffie-Helman
        if self.verbose:
            print("Send DH public key\n")

        dh_exchange(socket, public_key)

        # Recieve other devices key
        if self.verbose:
            print("Waiting for DH public key\n")

        other_public_key_bytes = dh_exchange_standby(socket, self.timeout)

        if other_public_key_bytes is None:
            if self.verbose:
                print("No initial key for Diffie-Helman recieved - early exit\n")
            return

        other_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            self.ec_curve, other_public_key_bytes
        )

        # Shared key generated
        shared_key = initial_private_key.exchange(ec.ECDH(), other_public_key)

        return shared_key

    def hash_function(self, bytes: bytes) -> bytes:
        """
        Generates a cryptographic hash of the given byte sequence using SHA-256.

        :param bytes: The data to hash.
        :return: The resulting hash as a byte string.
        """

        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def send_nonce_msg_to_device(
        self,
        connection: socket.socket,
        recieved_nonce_msg: bytes,
        derived_key: bytes,
        prederived_key_hash: bytes,
    ) -> bytes:
        """
        Sends a nonce message to a device, including a HMAC tag for verification.

        :param connection: The network connection object.
        :param received_nonce_msg: The nonce message received from the host.
        :param derived_key: The cryptographic key derived from an earlier exchange.
        :param prederived_key_hash: The hash of the key derived prior to this function.
        :return: The nonce generated in this function.
        """
        nonce = os.urandom(self.nonce_byte_size)

        # Concatenate nonces together
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len : pd_hash_len + self.nonce_byte_size
        ]
        concat_nonce = nonce + recieved_nonce

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(concat_nonce)
        tag = mac.finalize()

        # Construct nonce message
        nonce_msg = nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def send_nonce_msg_to_host(
        self, connection: socket.socket, prederived_key_hash: bytes, derived_key: bytes
    ) -> bytes:
        """
        Sends a nonce message to the host, including a HMAC tag for verification.

        :param connection: The network connection object.
        :param prederived_key_hash: The hash of the pre-derived key.
        :param derived_key: The cryptographic key derived from an earlier exchange.
        :return: The nonce generated in this function.
        """
        # Generate Nonce
        nonce = os.urandom(self.nonce_byte_size)

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(nonce)
        tag = mac.finalize()

        # Create key confirmation message
        nonce_msg = prederived_key_hash + nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def verify_mac_from_host(
        self, recieved_nonce_msg: bytes, generated_nonce: bytes, derived_key: bytes
    ) -> bool:
        """
        Verifies the HMAC tag received from the host.

        :param received_nonce_msg: The nonce message received from the host.
        :param generated_nonce: The nonce generated by the host, expected to match.
        :param derived_key: The cryptographic key used for HMAC.
        :return: True if the verification is successful, False otherwise.
        """
        success = False

        recieved_nonce = recieved_nonce_msg[0: self.nonce_byte_size]

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce + generated_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[self.nonce_byte_size:]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def verify_mac_from_device(
        self, recieved_nonce_msg: bytes, derived_key: bytes, prederived_key_hash: bytes
    ) -> bool:
        """
        Verifies the HMAC tag received from a device.

        :param received_nonce_msg: The nonce message received from the device.
        :param derived_key: The cryptographic key used for HMAC.
        :param prederived_key_hash: The hash of the pre-derived key.
        :return: True if the verification is successful, False otherwise.
        """
        success = False

        # Retrieve nonce used by device
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len: pd_hash_len + self.nonce_byte_size
        ]

        # Generate new MAC tag for the nonce with respect to the derived key
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[pd_hash_len + self.nonce_byte_size:]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success


"""###TESTING CODE###
import socket
def device(prot):
    print("device")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))
    prot.device_protocol(s)

def host(prot):
    print("host")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(0)
    prot.host_protocol([conn])

if __name__ == "__main__":
    import multiprocessing as mp
    from test_sensor import Test_Sensor
    from sensor_reader import Sensor_Reader
    prot = Miettinen_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024)),
        8,
        4,
        1,
        1,
        0.003,
        2.97e-05,
        0.9,
        5,
        20,
        10)
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()"""
