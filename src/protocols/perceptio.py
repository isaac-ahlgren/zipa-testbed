from typing import Any, List, Optional, Tuple

import numpy as np
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj
from networking.network import (
    ack,
    ack_standby,
    commit_standby,
    get_nonce_msg_standby,
    send_commit,
    send_status,
    socket,
    status_standby,
)
from protocols.common_protocols import (
    send_nonce_msg_to_device,
    send_nonce_msg_to_host,
    verify_mac_from_device,
    verify_mac_from_host,
)
from protocols.protocol_interface import ProtocolInterface
from signal_processing.perceptio import PerceptioProcessing


class Perceptio_Protocol(ProtocolInterface):
    def __init__(self, parameters: dict, sensor: Any, logger: any) -> None:
        """
        Implements a specific protocol to handle data communication and processing based on the ProtocolInterface.

        :param parameters: Dictionary containing various protocol-specific parameters.
        :param sensor: Sensor object used to collect data.
        :param logger: Logger object for logging various protocol activities and data.

        The protocol manages the initialization and execution of data processing tasks, using sensor input and specified parameters.
        It is capable of handling multiple iterations and configurations, making it suitable for experimental and operational environments.
        """

        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "Perceptio_Protocol"
        self.wip = False
        self.key_length = parameters["key_length"]
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.key_length + self.parity_symbols
        self.a = parameters["a"]
        self.cluster_sizes_to_check = parameters["cluster_sizes_to_check"]
        self.cluster_th = parameters["cluster_th"]
        self.top_th = parameters["top_th"]
        self.Fs = parameters["frequency"]
        self.bottom_th = parameters["bottom_th"]
        self.lump_th = parameters["lump_th"]
        self.conf_threshold = parameters["conf_thresh"]
        self.max_iterations = parameters["max_iterations"]
        self.min_events = parameters["min_events"]
        self.chunk_size = parameters["chunk_size"]
        self.nonce_byte_size = 16
        self.count = 0
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )

    def process_context(self) -> Any:
        """
        Processes the sensor data to detect events and generate cryptographic keys or fingerprints from them.

        :return: The fingerprints generated from the processed data.
        """
        events = []
        event_features = []
        iteration = 0
        while len(events) < self.min_events:
            chunk = self.read_samples(self.chunk_size)
            # Extracted from read_samples function in protocol_interface

            received_events = PerceptioProcessing.get_events(
                chunk, self.a, self.bottom_th, self.top_th, self.lump_th
            )

            received_event_features = PerceptioProcessing.get_event_features(
                received_events, chunk
            )

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

        ProtocolInterface.reset_flag(self.queue_flag)
        self.clear_queue()

        labels, k, _ = PerceptioProcessing.kmeans_w_elbow_method(
            event_features, self.cluster_sizes_to_check, self.cluster_th
        )

        grouped_events = PerceptioProcessing.group_events(events, labels, k)

        fps = PerceptioProcessing.gen_fingerprints(
            grouped_events, k, self.commitment_length, self.Fs
        )  # I know the variables are wrong, could someone fix them for me? -Isaac

        return fps

    #  TODO: Fix why this does not save correctly to drive
    def parameters(self, is_host: bool) -> str:
        pass

    def device_protocol(self, host_socket: socket.socket) -> None:
        """
        Conducts the device protocol over a given socket.

        :param host_socket: The socket connected to the host.
        This method handles the protocol's main loop including sending ACKs, extracting context,
        receiving commitments, and performing key confirmation until the desired number of successes
        is reached or the maximum number of iterations is exhausted. Logs the result of the protocol
        engagement including the number of successful key exchanges.
        """
        host_socket.setblocking(False)
        host_socket.settimeout(30)

        # self.name = "DEVICE"
        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        successes = 0
        iterations = 1
        while successes < self.conf_threshold and iterations <= self.max_iterations:
            success = False

            # Sending ack that they are ready to begin
            if self.verbose:
                print("[CLIENT] Sending ACK\n")
            ack(host_socket)

            if self.verbose:
                print("[CLIENT] Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("[CLIENT] No ACK recieved within time limit - early exit.\n")
                return

            if self.verbose:
                print("[CLIENT]  Extracting context\n")

            # Extract bits from sensor
            # witnesses, signal, status = self.extract_context(host_socket)
            witnesses = self.get_context()
            print(f"[CLIENT] Witnesses: {witnesses}\n")

            # TODO: Must log signal somehow, and how does status play with new flow?
            signal = None  # noqa:F841
            status = True  # noqa:F841

            """if status is None:
                if self.verbose:
                    print(
                        "Other device did not respond during extraction - early exit\n"
                    )
                return
            elif not status:
                if self.verbose:
                    print("Not enough events detected: moving to next iteration")
                continue"""

            if self.verbose:
                print("[CLIENT] Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)
            print(f"[CLIENT] Commitments: {commitments}\n")
            print(f"[CLIENT] Hashes: {hs}\n")

            # Early exist if no commitment recieved in time
            if not commitments:
                if self.verbose:
                    print(
                        "[CLIENT] No commitment recieved within time limit - early exit\n"
                    )
                return

            if self.verbose:
                print("[CLIENT] Uncommiting with witnesses\n")
            key = self.find_commitment(commitments, hs, witnesses)

            success = key is not None

            send_status(host_socket, success)

            # TODO: Fails to uncommit only sometimes which is weird
            # Commitment failed, try again
            if key is None:
                if self.verbose:
                    print(
                        "Witnesses failed to uncommit any commitment - alerting other device for a retry\n"
                    )
                # self.checkpoint_log(witnesses, commitments, success, signal, iterations)
                iterations += 1
                continue

            # Key Confirmation Phase

            if self.verbose:
                print("[CLIENT] Performing key confirmation\n")

            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(key)

            # Hash prederived key
            pd_key_hash = self.hash_function(key)

            if self.verbose:
                print("[CLIENT] Sending nonce message to host.\n")
            # Send nonce message to host
            generated_nonce = send_nonce_msg_to_host(
                host_socket,
                pd_key_hash,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            )

            if self.verbose:
                print("[CLIENT] Recieving nonce message from host.\n")
            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(
                host_socket, self.timeout
            )  # TODO Hangs here

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print(
                        "[CLIENT] No nonce message recieved within time limit - early exit\n"
                    )
                iterations += 1
                continue

            if self.verbose:
                print("[CLIENT] Comparing hashes.\n")
            # If hashes are equal, then it was successful
            if verify_mac_from_host(
                recieved_nonce_msg,
                generated_nonce,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            ):
                success = True
                successes += 1

            if self.verbose:
                print(
                    f"[CLIENT] Success? {success}, Successes: {successes}, Iterations: {iterations}\n"
                )

            # self.checkpoint_log(witnesses, commitments, success, signal, iterations)
            iterations += 1

        if self.verbose:
            print(
                f"[CLIENT] Total Key Pairing Result: success - {successes >= self.conf_threshold}\n"
            )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {successes} total_iterations: {iterations} succeeded: {successes >= self.conf_threshold}",
                )
            ]
        )

    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Manages the host-side protocol in a single-threaded manner using a specific device socket.

        :param device_socket: Socket connected to the client device.
        This method handles the host-side logic for establishing a secure connection
        and confirming keys with a single client. It oversees the entire protocol process
        including sending and receiving acknowledgments, extracting context, committing witnesses,
        and verifying keys until the desired success threshold is reached or the maximum iterations are exhausted.
        """
        # self.name="HOST"
        device_socket.setblocking(False)
        device_socket.settimeout(30)
        device_ip_addr, device_port = device_socket.getpeername()

        successes = 0
        iterations = 1
        while successes < self.conf_threshold and iterations <= self.max_iterations:
            success = False
            # Exit early if no devices to pair with
            if not ack_standby(device_socket, self.timeout):
                if self.verbose:
                    print("[HOST] No ACK recieved within time limit - early exit.\n")
                return
            if self.verbose:
                print("[HOST] Successfully ACKed participating device\n")

            if self.verbose:
                print("[HOST] ACKing all participating devices")
            ack(device_socket)

            if self.verbose:
                print("[HOST]  Extracting context\n")
            # Extract bits from sensor
            witnesses = self.get_context()
            print(f"[HOST] Witnesses: {witnesses}\n")

            signal = None  # noqa: F841
            status = True

            """if status is None:
                if self.verbose:
                    print(
                        "Other device did not respond during extraction - early exit\n"
                    )
                return
            elif not status:
                if self.verbose:
                    print("Not enough events detected: moving to next iteration")
                continue"""

            if self.verbose:
                print("[HOST] Commiting all the witnesses\n")
            # Create all commitments
            commitments, keys, hs = self.generate_commitments(witnesses)
            print(f"[HOST] Commitments: {commitments}\n")
            print(f"[HOST] Hashes: {hs}")

            if self.verbose:
                print("[HOST] Sending commitments\n")
            # Send all commitments

            send_commit(commitments, hs, device_socket)

            # Check up on other devices status
            status = status_standby(device_socket, self.timeout)  # TODO Hangs here
            if status is None:
                if self.verbose:
                    print("No status recieved within time limit - early exit.\n\n")
                return
            elif status is False:
                if self.verbose:
                    print(
                        "Other device did not uncommit with witnesses - trying again\n"
                    )
                success = False
                """self.checkpoint_log(
                    witnesses,
                    commitments,
                    success,
                    signal,
                    iterations,
                    ip_addr=device_ip_addr,
                )"""
                iterations += 1
                continue

            # Key Confirmation Phase

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(
                device_socket, self.timeout
            )  # TODO hangs here

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print(
                        "[HOST] No nonce message recieved within time limit - early exit\n"
                    )

                iterations += 1
                continue

            derived_key = self.host_verify_mac(keys, recieved_nonce_msg)

            if derived_key:
                success = True
                successes += 1

                # Create and send key confirmation value
                send_nonce_msg_to_device(
                    device_socket,
                    recieved_nonce_msg,
                    derived_key,
                    hs[0],
                    self.nonce_byte_size,
                    self.hash_func,
                )

                """self.checkpoint_log(
                    witnesses,
                    commitments,
                    success,
                    signal,
                    iterations,
                    ip_addr=device_ip_addr,
                )"""

            if self.verbose:
                print(
                    f"[HOST] Success? {success}, Successes: {successes}, Iterations: {iterations}\n"
                )

            iterations += 1

        if self.verbose:
            print(
                f"[HOST] Total Key Pairing Result: success - {successes >= self.conf_threshold}\n"
            )
        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {successes} total_iterations: {iterations} succeeded: {successes >= self.conf_threshold})",
                )
            ],
            ip_addr=device_ip_addr,
        )

    def host_verify_mac(
        self,
        keys: List[bytes],
        received_nonce_msg: bytes,
    ) -> Optional[bytes]:
        """
        Verifies the MAC received from a device against the derived keys.

        :param keys: List of keys to verify against.
        :param received_nonce_msg: The nonce message received from the device to verify.
        :return: The derived key if verification is successful, otherwise None.
        """
        key_found = None
        for i in range(len(keys)):
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(keys[i])
            key_hash = self.hash_function(keys[i])

            if verify_mac_from_device(
                received_nonce_msg,
                derived_key,
                key_hash,
                self.nonce_byte_size,
                self.hash_func,
            ):
                key_found = derived_key
                break
        return key_found

    def find_commitment(
        self, commitments: List[bytes], hashes: List[bytes], fingerprints: List[bytes]
    ) -> Optional[bytes]:
        """
        Attempts to match commitments with fingerprints based on their hash values.

        :param commitments: List of commitments to check against.
        :param hashes: Corresponding hashes of the commitments.
        :param fingerprints: Fingerprints to verify the commitments with.
        :return: The found key if a commitment is successfully matched, otherwise None.
        """
        key = None
        for i in range(len(fingerprints)):
            for j in range(len(commitments)):
                potential_key = self.re.decommit_witness(
                    commitments[j], fingerprints[i]
                )
                potential_key_hash = self.hash_function(potential_key)
                if constant_time.bytes_eq(potential_key_hash, hashes[j]):
                    key = potential_key
                    break
        return key

    def generate_commitments(
        self, witnesses: List[bytes]
    ) -> Tuple[List[bytes], List[bytes], List[bytes]]:
        """
        Generates commitments for a list of witnesses.

        :param witnesses: List of witnesses to generate commitments for.
        :return: A tuple containing the list of commitments, the keys used for commitments, and their hashes.
        """
        commitments = []
        keys = []
        hs = []
        for i in range(len(witnesses)):
            key, commitment = self.re.commit_witness(witnesses[i])
            commitments.append(commitment)
            keys.append(key)
            hs.append(self.hash_function(key))
        return commitments, keys, hs

    def checkpoint_log(
        self,
        witnesses: List[bytes],
        commitments: List[bytes],
        success: bool,
        signal: np.ndarray,
        iterations: int,
        ip_addr: Optional[str] = None,
    ) -> None:
        """
        Logs various protocol states and data at a checkpoint.

        :param witnesses: Witness data involved in the protocol.
        :param commitments: Commitments involved in the protocol.
        :param success: Success status of the protocol iteration.
        :param signal: The signal data associated with the current checkpoint.
        :param iterations: The iteration count at the checkpoint.
        :param ip_addr: Optional IP address associated with the log entry.
        """
        self.logger.log(
            [
                ("witness", "txt", witnesses),
                ("commitments", "txt", commitments),
                ("success", "txt", str(success)),
                ("signal", "csv", signal),
            ],
            count=iterations,
            ip_addr=ip_addr,
        )


def device(prot: Perceptio_Protocol) -> None:
    """
    Sets up a device-side communication for the protocol.

    :param prot: An instance of the Perceptio_Protocol class to manage the device side of the connection.
    This function configures the socket settings, connects to a specified server, and initiates the device protocol.
    """
    print("device")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))
    prot.device_protocol(s)


def host(prot: Perceptio_Protocol) -> None:
    """
    Sets up a host-side communication for the protocol.

    :param prot: An instance of the Perceptio_Protocol class to manage the host side of the connection.
    This function configures the server socket settings, accepts incoming connections, and initiates the host protocol.
    """
    print("host")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(0)
    prot.host_protocol([conn])


# TODO: Update test case with new protocol arguments
if __name__ == "__main__":
    import multiprocessing as mp

    from networking.nfs import NFSLogger
    from sensors.sensor_reader import Sensor_Reader
    from sensors.test_sensor import TestSensor

    prot = Perceptio_Protocol(
        Sensor_Reader(TestSensor(44100, 44100 * 50, 1024, signal_type="random")),
        8,
        4,
        44100 * 20,
        0.3,
        3,
        0.08,
        0.75,
        0.5,
        5,
        5,
        20,
        5,
        10,
        10,
        NFSLogger(None, None, None, None, None, 1, "./data"),
    )
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()
