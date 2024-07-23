import queue
import socket
from multiprocessing import Lock, Process, Queue, Value
from multiprocessing.shared_memory import SharedMemory
from typing import Any, List, Tuple

import numpy as np
from cryptography.hazmat.primitives import hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class ProtocolInterface:
    def __init__(self, parameters: dict, sensor: Any, logger: Any) -> None:
        """
        A base class for defining communication protocol interfaces that handle data processing and communication using a sensor.

        :param parameters: Dictionary of configuration parameters for the protocol.
        :param sensor: The sensor object that gathers the data.
        :param logger: Logger object for recording protocol-related information.
        """
        self.verbose = parameters["verbose"]
        self.sensor = sensor
        self.logger = logger
        self.queue = Queue()
        self.flag = Value("i", 0)
        self.shm_active = Value("i", 0)
        self.key_length = parameters["key_length"]
        self.time_length = None  # To be calculated in implementation
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.parity_symbols + self.key_length
        self.mutex = Lock()
        self.timeout = parameters["timeout"]
        self.hash_func = hashes.SHA256()
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )
        self.sensor.add_protocol_queue((self.flag, self.queue))

    def hash_function(self, bytes_data: bytes) -> bytes:
        """
        Computes a cryptographic hash of the given bytes.

        :param bytes: Bytes to be hashed.
        :return: The hash of the input bytes.
        """
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes_data)

        return hash_func.finalize()

    def host_protocol(self, device_sockets: List[socket.socket]) -> None:
        """
        Initiates the host protocol for each connected device socket in a separate process.

        :param device_sockets: A list of sockets, each connected to a different device.
        """
        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")

        processes = []

        for device in device_sockets:
            p = Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

    def get_signal(self):
        """
        Collects and returns signal data from a sensor until a specified time
        length is reached.

        This method reads data from a sensor queue, accumulating it until the
        time length condition is met. It starts collecting data when

        :return: List of collected signal data.
        """
        # Waiting until shared memory isn't needed by other processes
        while self.flag.value == -1:
            continue

        # First process to grab the flag populates the list
        if self.flag.value == 0:
            with self.mutex:
                signal = []
                self.flag.value = 1

                while self.flag.value == 1:
                    try:
                        data = self.queue.get()
                        signal.extend(data)
                    except queue.Empty:
                        continue

                    if len(signal) >= self.time_length:
                        shared_memory = SharedMemory(
                            name=self.name + "_Signal",
                            create=True,
                            size=self.sensor.sensor.data_type.itemsize
                            * self.time_length,
                        )
                        accessible_buffer = np.ndarray(
                            (self.time_length,),
                            self.sensor.sensor.data_type,
                            buffer=shared_memory.buf,
                        )
                        accessible_buffer[:] = signal[: self.time_length]
                        self.flag.value = -1
        # Remaining processes standy for list to be full
        else:
            while self.flag.value == 1:
                continue

            shared_memory = SharedMemory(
                name=self.name + "_Signal",
                create=False,
                size=self.sensor.sensor.data_type.itemsize * self.time_length,
            )
            accessible_buffer = np.ndarray(
                (self.time_length,),
                self.sensor.sensor.data_type,
                buffer=shared_memory.buf,
            )
        return list(accessible_buffer)

    # Must be implemented on a protocol basis
    def device_protocol(self, host: socket.socket) -> None:
        """
        Abstract method for device-side protocol logic. This method must be implemented by subclasses.

        :param host: The host connection or socket.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Abstract method for running the host protocol in a single-threaded manner for a single device. This method must be implemented by subclasses.

        :param device_socket: The socket connection to the host.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def extract_context(self) -> Tuple[bytes, List[float]]:
        """
        Abstract method for extracting relevant context or data during the protocol execution. This method must be implemented by subclasses.

        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def parameters(self, is_host: bool) -> str:
        """
        Abstract method for retrieving the parameters of the protocol. This method must be implemented by subclasses.

        :param is_host: Boolean indicating whether the parameters should be formatted for the host.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
