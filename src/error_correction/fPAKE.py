import os

import numpy as np
from cryptography import exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed448, x448
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from error_correction.liPAKE import LiPake

from networking.network import send_fpake_msg, fpake_msg_standby, send_status, status_standby, SUCC
from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self, pw_length, key_length,  timeout, so_dir=os.getcwd()):
        self.VK = None
        self.hash = hashes.SHA256()
        self.curve = x448
        self.ecpub = x448.X448PublicKey
        self.ed = ed448.Ed448PrivateKey
        self.edpub = ed448.Ed448PublicKey
        self.keySize = 32  # Keysize of the Curve448 public Key
        self.n = 32  # securityBits 256
        self.pw_length = pw_length
        self.key_length = key_length

        self.algo = algorithms.AES
        self.mode = modes.CBC
        self.hkdf = HKDFExpand(hashes.SHA3_256(), 32, None, default_backend())
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.pw_length, self.key_length, so_dir=os.getcwd()), self.key_length
        )
        self.timeout = timeout

    def generate_priv_key(self):
        return self.ed.generate()

    def generate_pub_key(self, priv_key):
        pub_key = priv_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        pub_key_plus_padding = pub_key + os.urandom(8)
        return pub_key_plus_padding
    
    def generate_symm_key(self, pw_i, vk):
        key = self.hkdf.derive(pw_i + vk)
        return key

    def encode(self, symm_enc_key, pub_key, iv):
        encryptor = Cipher(
            self.algo(symm_enc_key), self.mode(iv), default_backend()
        ).encryptor()

        return encryptor.update(pub_key) + encryptor.finalize()

    def decode(self, symm_dec_key, pub_key, iv):
        decryptor = Cipher(self.algo(symm_dec_key), self.mode(iv), default_backend()).decryptor()

        return decryptor.update(pub_key) + decryptor.finalize()

    def exchange(self, priv_key, star1, star2, Z):
        return priv_key
        

    def host_protocol(self, pw, conn):

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # Prepare for @LiPake exchange
        key_array = bytearray()
        # Execute @LiPake over each bit of the password
        for pw_i in pw:

            iv = os.urandom(
                16
            )  # We use fixed size of AES256 cause we can't get 128 Bit Keys with KDF
            priv_x = self.generate_priv_key()
            pub_x = self.generate_pub_key(priv_x)
            symm_enc_key = self.generate_symm_key(pw_i, vk)
            X_star = self.encode(symm_enc_key, pub_x, iv)

            # Send generated X receive Y
            send_fpake_msg(conn, [X_star, vk, iv])
            msg = fpake_msg_standby(conn, self.timeout)
            Y_star = msg[0]
            eps = msg[1]

            symm_dec_key = self.generate_symm_key(pw_i, eps)
            Z

            k = lp.getKey(Y, l, self.ecpub, False)
            key_array += k

        secret_key, commitment = self.re.commit_witness(key_array)
            
        # Sign our E with the secret key
        sig = signingKey.sign(commitment)

        # Send E + Signature + verification key + selected prime number to reconstruct
        send_fpake_msg(conn, [commitment, sig, vkBytes])

        status = status_standby(conn, self.timeout)

        if status != SUCC:
            secretkey = None

        return secretkey

    def device_protocol(self, pw, conn):
        labelList = []
        key_array = bytearray()
        # LiPake iteration for each bit in password
        for i in pw:
            # get Init Vectors as well as labels and X_s for LiPake
            msg = fpake_msg_standby(conn, self.timeout)
            Xs = msg[0]
            l1 = msg[1]
            iv = msg[2]

            if self.VK == None:
                try:
                    self.VK = self.edpub.from_public_bytes(l1)
                except:
                    print("VK was not correct ! Abort the protocol")
                    raise NotAVerificationKeyException
    
            labelList.append(l1)
            lp = LiPake(
                pw=i,
                label=b"",
                iv=iv,
                Hash=self.hash,
                mode=self.mode,
                curve=self.curve,
                symmAlgo=self.symmAlgo,
                symmetricKeySize=self.keySize,
                n=self.n,
            )
            # Generate receivers Y_s
            Ys, l2 = lp.getX()

            # Send Y_s with its label
            send_fpake_msg(conn, [Ys, l2])

            k = lp.getKey(Xs, l1, self.ecpub, True)
            key_array += k

        msg = fpake_msg_standby(conn, self.timeout)
        commitment = msg[0]
        sig = msg[1]
        vk = msg[2]

        # Reconstruct Verification key of bytes and verify the send E
        if vk != self.VK.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ):
            raise exceptions.InvalidKey()
        
        try:
            self.VK.verify(sig, commitment)
        except:  # Cancel if signature if wrong
            send_status(conn, False)
            return None

        secret_key = self.re.decommit_witness(key_array, commitment)

        send_status(conn, True)

        return secret_key


class NotAVerificationKeyException(Exception):
    pass
