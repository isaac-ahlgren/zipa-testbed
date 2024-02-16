from cryptography.hazmat.primitives import hashes
import random
import numpy as np

class Fuzzy_Commitment:
    def __init__(self, error_correction_obj, key_byte_length):
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj
    
    def commit_witness(self, witness, use_hash_func=True):
        # Generate secret key
        secret_key = random.randbytes(self.key_byte_length)

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        # Commit witness by getting XOR distance between codeword and witness
        C ^= witness
        return secret_key, C

    def decommit_witness(self, C, witness):
        C ^= witness

        secret_key = self.error_correction.decode(C)

        return secret_key
