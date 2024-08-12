import os

import numpy as np
from cryptography import exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed448, x448
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from networking.network import send_fpake_msg, fpake_msg_standby, send_status, status_standby
from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self, pw_length, key_length,  timeout, so_dir=os.getcwd()):
        self.hash = hashes.SHA256()
        self.curve = x448
        self.ec = x448.X448PrivateKey
        self.ecpub = x448.X448PublicKey
        self.ed = ed448.Ed448PrivateKey
        self.edpub = ed448.Ed448PublicKey
        self.pw_length = pw_length
        self.key_length = key_length
        self.hash = hashes.SHA3_256()
        self.algo = algorithms.AES
        self.mode = modes.CBC
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.pw_length*32, self.key_length*32, so_dir=os.getcwd()), self.key_length*32
        )
        self.timeout = timeout

    def generate_priv_key(self):
        return self.ec.generate()

    def generate_pub_key(self, priv_key):
        pub_key = priv_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        pub_key_plus_padding = pub_key + os.urandom(8)
        return pub_key_plus_padding
    
    def generate_symm_key(self, pw_i, vk):
        hkdf = HKDFExpand(self.hash, 32, None, default_backend())
        key = hkdf.derive(pw_i + vk)
        return key

    def encode(self, symm_enc_key, pub_key, iv):
        encryptor = Cipher(
            self.algo(symm_enc_key), self.mode(iv), default_backend()
        ).encryptor()

        return encryptor.update(pub_key) + encryptor.finalize()

    def decode(self, symm_dec_key, enc_pub_key, priv_key, iv):
        decryptor = Cipher(self.algo(symm_dec_key), self.mode(iv), default_backend()).decryptor()
        dec_pub_key = decryptor.update(enc_pub_key) + decryptor.finalize()
        pub_key = self.ecpub.from_public_bytes(dec_pub_key[0:56])
        return priv_key.exchange(pub_key)

    def gen_key_part(self, star1, star2, Z):
        hash_func = hashes.Hash(self.hash, default_backend())
        hash_func.update(star1)
        hash_func.update(star2)
        hash_func.update(Z)
        return hash_func.finalize()
        

    def host_protocol(self, pw, conn):

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        send_fpake_msg(conn, [vk])

        key_array = bytearray()
        for pw_i in pw:
            pw_i = pw_i.to_bytes(1, "big")

            iv = os.urandom(
                16
            )  # We use fixed size of AES256 cause we can't get 128 Bit Keys with KDF
            priv_x = self.generate_priv_key()
            pub_x = self.generate_pub_key(priv_x)
            symm_enc_key = self.generate_symm_key(pw_i, vk)
            X_star = self.encode(symm_enc_key, pub_x, iv)

            # Send generated X receive Y
            send_fpake_msg(conn, [X_star, iv])
            msg = fpake_msg_standby(conn, self.timeout)
            Y_star = msg[0]
            eps = msg[1]

            symm_dec_key = self.generate_symm_key(pw_i, eps)
            
            Z = self.decode(symm_dec_key, Y_star, priv_x, iv)

            k = self.gen_key_part(X_star, Y_star, Z)
            key_array += k

        secret_key, commitment = self.re.commit_witness(key_array)

        # Sign our E with the secret key
        sig = signingKey.sign(commitment)

        # Send E + Signature + verification key + selected prime number to reconstruct
        send_fpake_msg(conn, [commitment, sig, vk])

        status = status_standby(conn, self.timeout)

        if not status:
            secret_key = None

        return secret_key

    def device_protocol(self, pw, conn):
        signingKey = self.ed.generate()
        eps = signingKey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        msg = fpake_msg_standby(conn, self.timeout)
        vk = msg[0]

        try:
            self.edpub.from_public_bytes(vk)
        except:
            print("VK was not correct ! Abort the protocol")
            raise NotAVerificationKeyException

        labelList = []
        key_array = bytearray()
        # LiPake iteration for each bit in password
        for pw_i in pw:
            pw_i = pw_i.to_bytes(1, "big")

            priv_y = self.generate_priv_key()
            pub_y = self.generate_pub_key(priv_y)
            symm_enc_key = self.generate_symm_key(pw_i, eps)

            # get Init Vectors as well as labels and X_s for LiPake
            msg = fpake_msg_standby(conn, self.timeout)
            X_star = msg[0]
            iv = msg[1]

            Y_star = self.encode(symm_enc_key, pub_y, iv)

            symm_dec_key = self.generate_symm_key(pw_i, vk)

            Z = self.decode(symm_dec_key, X_star, priv_y, iv)

            # Send Y_s with its label
            send_fpake_msg(conn, [Y_star, eps])

            k = self.gen_key_part(X_star, Y_star, Z)
            key_array += k

        msg = fpake_msg_standby(conn, self.timeout)
        commitment = msg[0]
        sig = msg[1]
        
        try:
            vk_key =  self.edpub.from_public_bytes(vk)
            vk_key.verify(sig, commitment)
        except:  # Cancel if signature if wrong
            send_status(conn, False)
            return None

        secret_key = self.re.decommit_witness(key_array, commitment)

        send_status(conn, True)

        return secret_key


class NotAVerificationKeyException(Exception):
    pass
