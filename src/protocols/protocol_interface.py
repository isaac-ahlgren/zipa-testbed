import queue
import socket
from multiprocessing import Lock, Process, Queue, Value
from multiprocessing.shared_memory import ShareableList
from typing import Any, List, Tuple

import numpy as np
from cryptography.hazmat.primitives import hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj

READY = 0
PROCESSING = 1
COMPLETE = -1


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
        self.queue_flag = Value("i", 0)
        self.processing_flag = Value("i", 0)
        self.shm_active = Value("i", 0)
        self.key_length = parameters["key_length"]
        self.time_length = None  # To be calculated in implementation
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.parity_symbols + self.key_length
        self.mutex = Lock()
        self.timeout = parameters["timeout"]
        self.hash_func = hashes.SHA256()

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

    def capture_flag(shared_value: Any) -> None:
        """
        Grabs the shared value and sets it to `1`, meaning that the queue is
        ready for data collection.
        """
        shared_value.value = 1

    def release_flag(shared_value: Any) -> None:
        """
        Grabs the shared value and sets it to `-1`, meaning that the queue
        is populated with data and can be shared across processes
        """
        shared_value.value = -1

    def reset_flag(shared_value: Any) -> None:
        """
        Grabs the shared value and sets it to `0`, meaning that the queue
        is ready to be captured again by a process to accumulate data
        """
        shared_value.value = 0

    def write_shm(self, byte_list: List[bytes]) -> None:
        """
        This function should write the bits generated to shared memory,
        allowing other processes access

        :returns: bytes, from data sent into protocol algorithm
        """
        print("Writing to shared memory...")
        ShareableList(
            name=self.name + "_Bytes",
            sequence=byte_list,
        )

    def read_shm(self) -> List[bytes]:
        """
        To be used by other processes to retrieve bits that were already
        generated by another process
        """
        print("Reading from shared memory...")
        shared_list = ShareableList(name=self.name + "_Bytes")

        return shared_list

    def destroy_shm(self) -> None:
        """
        Destroys shared list reference, allowing namespace to be used
        again
        """
        shared_list = ShareableList(name=self.name + "_Bytes")
        shared_list.shm.unlink()

    def get_context(self) -> Any:
        """
        Manages the shared list usage for retrieving context data.
        """
        print("Entering get_context()") #ADDED
        results = None
        # Keep track if shared list is being used

        while self.processing_flag.value == COMPLETE and self.shm_active.value != 0:
            print("Waiting for shm to be free...")
            continue
        print("Accessing shared memory...")
        self.shm_active.value += 1
        # First process to grab the flag populates the list
        if self.processing_flag.value == READY:
            with self.mutex:
                print("Populating shared memory...")
                ProtocolInterface.capture_flag(self.processing_flag)
                try:
                    self.destroy_shm()
                except FileNotFoundError:
                    pass  # No shared memory instance to destroy
                results = self.process_context()
                self.write_shm(results)
                ProtocolInterface.release_flag(self.processing_flag)

        # Other processes wait for first process to finish
        else:
            while self.processing_flag.value == PROCESSING:
                print("Waiting for data to be ready...")
                continue

            results = self.read_shm()

        # Process no longer is using the shared list
        self.shm_active.value -= 1
        print("Leaving get_context()")
        return results

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

    def process_context(self) -> Any:
        """
        Processes the collected data. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def read_samples(self, sample_num: int) -> np.ndarray:
        """
        Reads specified number of samples from the queue.

        :param sample_num: The number of samples to read.
        :return: An array of the collected samples.
        """
        # Assuming only one process is handling this.
        samples_read = 0
        output = np.array([])
        # Signal status_queue is ready for data
        ProtocolInterface.capture_flag(self.queue_flag)

        while samples_read < sample_num:
            chunk = self.queue.get()
            output = np.append(output, chunk)
            samples_read += len(chunk)

        # TODO Must be implemented on a protocol basis
        # Signal status_queue doesn't need any more data
        # ProtocolInterface.reset_flag(self.queue_flag)
        # self.clear_queue()

        return output[:sample_num]

    def clear_queue(self) -> None:
        """
        Clears the status queue to assure fresh data the next time it's used.
        """
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

    def parameters(self, is_host: bool) -> str:
        """
        Abstract method for retrieving the parameters of the protocol. This method must be implemented by subclasses.

        :param is_host: Boolean indicating whether the parameters should be formatted for the host.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
