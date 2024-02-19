from cryptography.hazmat.primitives import hashes
import random
import numpy as np

class Fuzzy_Commitment:
    def __init__(self, error_correction_obj, key_byte_length):
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj
    
    def xor_bytes(self, bytes1: bytearray, bytes2: bytearray) -> bytearray:
        print(len(bytes1))
        print(len(bytes2))
        output = bytearray([0 for i in range(len(bytes1))])
        for i in range(len(bytes1)):
            output = bytes1[i] ^ bytes2[i]
        return output

    def commit_witness(self, witness):
        # Generate secret key
        secret_key = random.randbytes(self.key_byte_length)

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        # Commit witness by getting XOR distance between codeword and witness
        commitment = self.xor_bytes(C, witness)
        return secret_key, C

    def decommit_witness(self, commitment, witness):
        C = xor_bytes(commitment, witness)

        secret_key = self.error_correction.decode(C)

        return secret_key
