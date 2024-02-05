from cryptography.hazmat.primitives import hashes
import random

PRIME_POLY = 0b100011101
GEN_POLY =   0b10000011
BLOCK_SIZE = 8

class Fuzzy_Commitment:
    def __init__(self, error_correction_obj, key_byte_length):
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj
        self.hash = hashes.Hash(hashes.SHA512())
    
    def commit_witness(self, witness, use_hash_func=True):
        # Generate secret key
        secret_key = random.randbytes(self.key_byte_length)

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        h = None
        if use_hash_func:
            # Get hash for codeword
            h_func = hashes.Hash(hashes.SHA512())
            h_func.update(secret_key)
            h = h_func.finalize()

        # Commit witness by getting XOR distance between codeword and witness
        C ^= witness
        return key, h, C

    def decommit_witness(self, C, witness, h, use_hash_func=True):
        C ^= witness

        secret_key = self.error_correction.decode(C)

        sucess = None
        if use_hash_func:
            # Hashing corrected codeword and checking if pairing is a success
            h_func = hashes.Hash(hashes.SHA512())
            h_func.update(secret_key)
            check_h = h_func.finalize()
            if check_h == h:
                success = True
            else:
                success = False

        return secret_key, success
