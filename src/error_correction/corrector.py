# import random
import secrets
from typing import Optional, Tuple

# import numpy as np
# from cryptography.hazmat.primitives import hashes


class Fuzzy_Commitment:
    """
    Implements a fuzzy commitment scheme which allows one to commit to a chosen value (witness)
    while keeping it hidden to others. The commitment can later be opened only with the exact same
    witness or one that is close enough in terms of Hamming distance.

    :param error_correction_obj: An object that handles encoding and decoding which allows error correction.
    :type error_correction_obj: Any object with `encode` and `decode` methods compatible with the desired error correction scheme.
    :param key_byte_length: Length of the key in bytes.
    :type key_byte_length: int
    """

    def __init__(self, error_correction_obj: object, key_byte_length: int) -> None:
        """
        Initializes the fuzzy commitment scheme with an error correction mechanism and a specified key byte length.

        :param error_correction_obj: An error correction instance capable of encoding and decoding.
        :param key_byte_length: The byte length of the keys used in the scheme.
        """
        self.key_byte_length = key_byte_length
        self.error_correction = error_correction_obj

    def xor_bytes(self, bytes1: bytearray, bytes2: bytearray) -> bytearray:
        """
        Performs an XOR operation between two byte arrays of the same length.

        :param bytes1: The first byte array.
        :type bytes1: bytearray
        :param bytes2: The second byte array.
        :type bytes2: bytearray
        :returns: The result of byte-wise XOR between `bytes1` and `bytes2`.
        :rtype: bytearray
        """
        output = bytearray([0 for i in range(len(bytes1))])
        for i in range(len(bytes2)):
            output[i] = bytes2[i] ^ bytes1[i]
        return output

    def commit_witness(self, witness: bytearray) -> Tuple[bytes, bytearray]:
        """
        Commits a witness value by encoding and then XORing it with a randomly generated secret key.

        :param witness: The witness value to commit to, represented as a bytearray.
        :type witness: bytearray
        :returns: A tuple containing the secret key and the commitment.
        :rtype: tuple
        """
        # Generate secret key
        # secret_key = random.randbytes(self.key_byte_length)
        secret_key = secrets.token_bytes(self.key_byte_length)

        # Encode secret key
        C = self.error_correction.encode(secret_key)

        # Commit witness by getting XOR distance between codeword and witness
        commitment = self.xor_bytes(C, witness)
        return secret_key, commitment

    def decommit_witness(
        self, commitment: bytearray, witness: bytearray
    ) -> Optional[bytes]:
        """
        Decommit the witness using the commitment and the original witness value.

        :param commitment: The commitment produced by `commit_witness`.
        :type commitment: bytearray
        :param witness: The original witness value.
        :type witness: bytearray
        :returns: The secret key if successful, raises an error otherwise.
        :rtype: bytearray
        """
        # Decommit by XORing the commitment and the witness
        C = self.xor_bytes(commitment, witness)

        # Correct errors in the decommiting process due to the witness
        secret_key = self.error_correction.decode(C)

        return secret_key
