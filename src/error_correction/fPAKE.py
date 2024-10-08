import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import constant_time, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed448, x448
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from error_correction.corrector import Fuzzy_Commitment
from error_correction.simple_reed_solomon import SimpleReedSolomonObj
from networking.network import (
    pake_msg_standby,
    send_pake_msg,
    send_status,
    status_standby,
)


class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self, pw_length, key_length, timeout):
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
            SimpleReedSolomonObj(
                self.pw_length * 16,
                self.key_length * 16,
                power_of_2=16,
                prime_poly=0x11085,
            ),
            self.key_length * 32,
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
        decryptor = Cipher(
            self.algo(symm_dec_key), self.mode(iv), default_backend()
        ).decryptor()
        dec_pub_key = decryptor.update(enc_pub_key) + decryptor.finalize()
        pub_key = self.ecpub.from_public_bytes(dec_pub_key[0:56])
        return priv_key.exchange(pub_key)

    def gen_key_part(self, star1, star2, Z):
        hash_func = hashes.Hash(self.hash, default_backend())
        hash_func.update(star1)
        hash_func.update(star2)
        hash_func.update(Z)
        return hash_func.finalize()

    def derive(self, secret_key):
        kdf = PBKDF2HMAC(
            algorithm=self.hash,
            salt=b"",
            length=self.key_length,
            iterations=480000,
        )
        return kdf.derive(secret_key)

    def hash_function(self, bytes_data):
        hash_func = hashes.Hash(self.hash)
        hash_func.update(bytes_data)

        return hash_func.finalize()

    def compare(self, bytes1, bytes2):
        success = False
        if constant_time.bytes_eq(bytes1, bytes2):
            success = True
        return success

    def host_protocol(self, pw, conn):

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        send_pake_msg(conn, [vk])

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
            send_pake_msg(conn, [X_star, iv])
            msg = pake_msg_standby(conn, self.timeout)
            Y_star = msg[0]
            eps = msg[1]

            symm_dec_key = self.generate_symm_key(pw_i, eps)

            Z = self.decode(symm_dec_key, Y_star, priv_x, iv)

            k = self.gen_key_part(X_star, Y_star, Z)
            key_array += k

        secret_key, commitment = self.re.commit_witness(key_array)

        hash_val_0 = self.hash_function(secret_key + bytes(0))
        hash_val_1 = self.hash_function(secret_key + bytes(1))

        # Sign our E with the secret key
        sig = signingKey.sign(bytes(commitment))

        # Send E + Signature + verification key + selected prime number to reconstruct
        send_pake_msg(conn, [commitment, sig, hash_val_1])

        msg = pake_msg_standby(conn, self.timeout)
        other_hash_val = msg[0]

        other_success = status_standby(conn, self.timeout)

        success = self.compare(hash_val_0, other_hash_val) and other_success

        send_status(conn, success)

        if success:
            secret_key = self.derive(secret_key)
        else:
            secret_key = None

        return secret_key

    def device_protocol(self, pw, conn):
        signingKey = self.ed.generate()
        eps = signingKey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        msg = pake_msg_standby(conn, self.timeout)
        vk = msg[0]

        try:
            self.edpub.from_public_bytes(vk)
        except Exception:
            print("VK was not correct ! Abort the protocol")
            raise NotAVerificationKeyException

        key_array = bytearray()
        # LiPake iteration for each bit in password
        for pw_i in pw:
            pw_i = pw_i.to_bytes(1, "big")

            priv_y = self.generate_priv_key()
            pub_y = self.generate_pub_key(priv_y)
            symm_enc_key = self.generate_symm_key(pw_i, eps)

            # get Init Vectors as well as labels and X_s for LiPake
            msg = pake_msg_standby(conn, self.timeout)
            X_star = msg[0]
            iv = msg[1]

            Y_star = self.encode(symm_enc_key, pub_y, iv)

            symm_dec_key = self.generate_symm_key(pw_i, vk)

            Z = self.decode(symm_dec_key, X_star, priv_y, iv)

            # Send Y_s with its label
            send_pake_msg(conn, [Y_star, eps])

            k = self.gen_key_part(X_star, Y_star, Z)
            key_array += k

        msg = pake_msg_standby(conn, self.timeout)
        commitment = msg[0]
        sig = msg[1]
        other_hash_val = msg[2]

        success = None
        try:
            vk_key = self.edpub.from_public_bytes(vk)
            vk_key.verify(sig, commitment)
            success = True
        except Exception:  # Cancel if signature if wrong
            success = False

        secret_key = self.re.decommit_witness(key_array, commitment)

        hash_val_0 = self.hash_function(secret_key + bytes(0))
        hash_val_1 = self.hash_function(secret_key + bytes(1))

        success = success and self.compare(hash_val_1, other_hash_val)

        send_pake_msg(conn, [hash_val_0])
        send_status(conn, success)

        other_success = status_standby(conn, self.timeout)

        success = success and other_success

        if success:
            secret_key = self.derive(secret_key)
        else:
            secret_key = None

        return secret_key


class NotAVerificationKeyException(Exception):
    pass
