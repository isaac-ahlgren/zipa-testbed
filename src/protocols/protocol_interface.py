import multiprocessing as mp

from cryptography.hazmat.primitives import hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class ProtocolInterface:
    def __init__(self, parameters, logger):
        self.verbose = parameters["verbose"]
        self.sensor = parameters["sensor"]
        self.logger = logger
        self.key_length = parameters["key_length"]
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.parity_symbols + self.key_length
        self.timeout = parameters["timeout"]
        self.hash_func = hashes.SHA256()
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)

        return hash_func.finalize()

    def host_protocol(self, device_sockets):
        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    # Must be implemented on a protocol basis
    def device_protocol(self, host):
        raise NotImplementedError

    def host_protocol_single_threaded(self, device_socket):
        raise NotImplementedError
    
    def extract_context(self):
        raise NotImplementedError
    
    def parameters(self, is_host):
        raise NotImplementedError
