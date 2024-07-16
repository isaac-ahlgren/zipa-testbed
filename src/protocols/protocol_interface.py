from multiprocessing import Process, Queue, Value, Lock
from typing import List, Tuple, Any, Union

from cryptography.hazmat.primitives import hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class ProtocolInterface:
    def __init__(self, parameters: dict, sensor: Any, logger: Any) -> None:
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

    def hash_function(self, bytes_data: bytes) -> bytes:
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)

        return hash_func.finalize()

    def host_protocol(self, device_sockets: List[Any]) -> None:
        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")
        for device in device_sockets:
            p = Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    # Must be implemented on a protocol basis
    def device_protocol(self, host: Any) -> None:
        raise NotImplementedError

    def host_protocol_single_threaded(self, device_socket: Any) -> None:
        raise NotImplementedError

    def extract_context(self) -> Tuple[bytes, Any]:
        raise NotImplementedError

    def parameters(self, is_host: bool) -> str:
        raise NotImplementedError
