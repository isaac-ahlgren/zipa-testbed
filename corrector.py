from cryptography.hazmat.primitives import hashes
import random
import numpy as np

class Fuzzy_Commitment:
    def __init__(self, error_correction_obj, key_byte_length):
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj
        self.hash = hashes.Hash(hashes.SHA512())
    
    #TODO: Move use of hash function outside of this class (and into shurmann.py) not every protocol relies on a hash
    def commit_witness(self, witness, use_hash_func=True):
        # Generate secret key
        secret_key = random.randbytes(self.key_byte_length)

        h = None
        if use_hash_func:
            # Get hash for codeword
            self.hash.update(secret_key)
            h = self.hash.finalize()

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        # Commit witness by getting XOR distance between codeword and witness
        C ^= witness
        return secret_key, h, C

    def decommit_witness(self, C, witness, h, use_hash_func=True):
        C ^= witness

        secret_key = self.error_correction.decode(C)

        sucess = None
        if use_hash_func:
            # Hashing corrected codeword and checking if pairing is a success
            self.hash.update(secret_key)
            check_h = self.hash.finalize()
            if check_h == h:
                success = True
            else:
                success = False

        return secret_key, success
