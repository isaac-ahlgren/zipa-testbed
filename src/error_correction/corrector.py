# import random
import secrets
from typing import Tuple, Optional

import numpy as np
from cryptography.hazmat.primitives import hashes


class Fuzzy_Commitment:
    def __init__(self, error_correction_obj: object, key_byte_length: int) -> None:
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj

    def xor_bytes(self, bytes1: bytearray, bytes2: bytearray) -> bytearray:
        output = bytearray([0 for i in range(len(bytes1))])
        for i in range(len(bytes1)):
            output[i] = bytes1[i] ^ bytes2[i]
        return output

    def commit_witness(self, witness: bytearray) -> Tuple[bytes, bytearray]:
        # Generate secret key
        # secret_key = random.randbytes(self.key_byte_length)
        secret_key = secrets.token_bytes(self.key_byte_length)

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        # Commit witness by getting XOR distance between codeword and witness
        commitment = self.xor_bytes(C, witness)
        return secret_key, commitment

    def decommit_witness(self, commitment: bytearray, witness: bytearray) -> Optional[bytes]:
        # Decommit by XORing the commitment and the witness
        C = self.xor_bytes(commitment, witness)

        # Correct errors in the decommiting process due to the witness
        secret_key = self.error_correction.decode(C)

        return secret_key
