from multiprocessing import Process, Queue, Value, Lock

from cryptography.hazmat.primitives import hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class ProtocolInterface:
    """
    A base class for defining communication protocol interfaces that handle data processing and communication using a sensor.

    :param parameters: Dictionary of configuration parameters for the protocol.
    :param sensor: The sensor object that gathers the data.
    :param logger: Logger object for recording protocol-related information.
    """
    def __init__(self, parameters, sensor, logger):
        self.verbose = parameters["verbose"]
        self.sensor = sensor
        self.logger = logger
        self.queue = Queue()
        self.flag = Value("i", 0)
        self.key_length = parameters["key_length"]
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.parity_symbols + self.key_length
        self.mutex = Lock()
        self.timeout = parameters["timeout"]
        self.hash_func = hashes.SHA256()
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )
        self.sensor.add_protocol_queue((self.flag, self.queue))

    def hash_function(self, bytes):
        """
        Computes a cryptographic hash of the given bytes.

        :param bytes: Bytes to be hashed.
        :return: The hash of the input bytes.
        """
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)

        return hash_func.finalize()

    def host_protocol(self, device_sockets):
        """
        Initiates the host protocol for each connected device socket in a separate process.

        :param device_sockets: A list of sockets, each connected to a different device.
        """
        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")
        for device in device_sockets:
            p = Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    # Must be implemented on a protocol basis
    def device_protocol(self, host):
        """
        Abstract method for device-side protocol logic. This method must be implemented by subclasses.

        :param host: The host connection or socket.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def host_protocol_single_threaded(self, device_socket):
        """
        Abstract method for running the host protocol in a single-threaded manner for a single device. This method must be implemented by subclasses.

        :param device_socket: The socket connection to the host.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def extract_context(self):
        """
        Abstract method for extracting relevant context or data during the protocol execution. This method must be implemented by subclasses.

        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def parameters(self, is_host):
        """
        Abstract method for retrieving the parameters of the protocol. This method must be implemented by subclasses.

        :param is_host: Boolean indicating whether the parameters should be formatted for the host.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
