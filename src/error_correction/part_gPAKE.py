import os
import json  # Import to handle session_id serialization
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import padding

from networking.network import (
    pake_msg_standby,
    send_pake_msg,
    send_status,
    status_standby,
)

class GPAKE:
    """
    Group Password Authenticated Key Exchange (GPAKE) Protocol
    """

    def __init__(self, curve=ec.SECP256R1(), timeout=30):
        # Initialize elliptic curve parameters
        self.curve = curve
        self.timeout = timeout
        self.hash = hashes.SHA256()
        self.algo = algorithms.AES
        self.mode = modes.CBC

    def generate_priv_key(self):
        """Generate a private key for EC Diffie-Hellman."""
        return ec.generate_private_key(self.curve, default_backend())

    def generate_pub_key(self, priv_key):
        """Generate a public key from the private key."""
        return priv_key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
    
    def generate_shared_key_host(self, priv_key, peer_pub_key_bytes, local_device_id, peer_device_id):
        """Generate shared key from the host's private key and the device's public key."""
        peer_pub_key = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, peer_pub_key_bytes)
        shared_key = priv_key.exchange(ec.ECDH(), peer_pub_key)

        info = (local_device_id.encode() + peer_device_id.encode() +
                priv_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint) +
                peer_pub_key_bytes)

        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        ).derive(shared_key)

        return derived_key
    
    def generate_shared_key_device(self, priv_key, peer_pub_key_bytes, local_device_id, peer_device_id):
        """Generate shared key from the device's private key and the host's public key."""
        peer_pub_key = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, peer_pub_key_bytes)
        shared_key = priv_key.exchange(ec.ECDH(), peer_pub_key)

        info = (peer_device_id.encode() + local_device_id.encode() +
                peer_pub_key_bytes +
                priv_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))

        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        ).derive(shared_key)

        return derived_key

    def encode(self, symm_key, pub_key_bytes, iv):
        """Encrypt the public key with symmetric encryption."""
        padder = padding.PKCS7(128).padder()  # Block size is 128 bits (16 bytes)
        padded_data = padder.update(pub_key_bytes) + padder.finalize()

        encryptor = Cipher(self.algo(symm_key), self.mode(iv), default_backend()).encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    def decode(self, symm_key, enc_pub_key, iv):
        """Decrypt the encrypted public key."""
        decryptor = Cipher(self.algo(symm_key), self.mode(iv), default_backend()).decryptor()
        decrypted_data = decryptor.update(enc_pub_key) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()  # Block size is 128 bits (16 bytes)
        unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
        return unpadded_data

    def hash_function(self, bytes_data):
        """Hash function to generate hashes of messages."""
        digest = hashes.Hash(self.hash, default_backend())
        digest.update(bytes_data)
        return digest.finalize()

    def compare(self, data1, data2):
        """Constant time comparison for secure equality check."""
        return data1 == data2

    def generate_key_pairs(self, grouped_events):
        """Generate private/public key pairs for each event type."""
        priv_keys = [self.generate_priv_key() for _ in grouped_events]
        pub_keys = [self.generate_pub_key(priv_key) for priv_key in priv_keys]
        return priv_keys, pub_keys
    
    def send_encrypted_public_key(self, conn, device_id, session_idx, pub_key, pw):
        """Encrypt and send the public key with device identifier and session ID."""
        iv = os.urandom(16)
        assert len(iv) == 16, "IV is not 16 bytes before sending"
        symm_key = self.hash_function(pw)
        enc_pub_key = self.encode(symm_key, pub_key, iv)
        enc_pub_key_b64 = base64.b64encode(enc_pub_key).decode()

        session_id = {
            'session_idx': session_idx, 
            'host_device_id': device_id, 
            'host_enc_pub_key': enc_pub_key_b64
        }
        session_id_bytes = json.dumps(session_id).encode()

        send_pake_msg(conn, [device_id.encode(), session_id_bytes, enc_pub_key, iv])

    def handle_device_public_key_exchange(self, conn, priv_key, password, device_id):
        """Handle receiving and decrypting device's public key and generate shared key."""
        msg = pake_msg_standby(conn, self.timeout)
        device_id_received = msg[0].decode()
        session_id_received = json.loads(msg[1].decode())
        enc_device_pub_key = msg[2]
        iv_received = msg[3]

        assert len(iv_received) == 16, "IV is not 16 bytes after receiving"

        symm_key = self.hash_function(password)
        device_pub_key_bytes = self.decode(symm_key, enc_device_pub_key, iv_received)
    
        return self.generate_shared_key_device(priv_key, device_pub_key_bytes, device_id, device_id_received)
    
    def generate_and_send_random_value(self, conn, shared_key, session_idx):
        """Generate random value, encrypt it with shared key, and send it."""
        random_value = os.urandom(16)
        iv_random = os.urandom(16)
        encrypted_random_value = self.encode(shared_key, random_value, iv_random)

        # Send the encrypted random value along with session ID and IV
        session_id_bytes = json.dumps({'session_idx': session_idx}).encode()
        send_pake_msg(conn, [session_id_bytes, encrypted_random_value, iv_random])

        return random_value
    
    def receive_and_decrypt_random_value(self, conn, shared_key):
        """Receive and decrypt the random value sent by the device."""
        msg = pake_msg_standby(conn, self.timeout)
        if msg is None:
            raise ValueError("No message received during random value decryption.")
        enc_random_value_received = msg[1]
        iv_received = msg[2]
        assert len(iv_received) == 16, f"Received IV is not 16 bytes, instead {len(iv_received)} bytes"

        decrypted_random_value = self.decode(shared_key, enc_random_value_received, iv_received)

        return decrypted_random_value
    
    def compute_final_key(self, random_values):
        """Sum all random values and hash the result to generate the final key."""
        final_group_key = b''.join(random_values)
        return self.hash_function(final_group_key)
    
    def handle_host_public_key_exchange(self, conn, priv_key, pub_key, password, device_id, session_idx):
        """Handle receiving and decrypting host's public key and send device's public key."""
        msg = pake_msg_standby(conn, self.timeout)
        host_device_id = msg[0].decode()
        session_id_received = json.loads(msg[1].decode())
        enc_host_pub_key = msg[2]
        iv_received = msg[3]

        assert len(iv_received) == 16, "IV is not 16 bytes after receiving"
        symm_key = self.hash_function(password)
        host_pub_key_bytes = self.decode(symm_key, enc_host_pub_key, iv_received)

        # Send device's encrypted public key to host
        iv = os.urandom(16)
        enc_device_pub_key = self.encode(symm_key, pub_key, iv)

        session_id = {
            'session_idx': session_idx,
            'device_id': device_id,
            'host_device_id': host_device_id,
            'device_enc_pub_key': base64.b64encode(enc_device_pub_key).decode(),
            'host_enc_pub_key': session_id_received['host_enc_pub_key']
        }
        session_id_bytes = json.dumps(session_id).encode()
        send_pake_msg(conn, [device_id.encode(), session_id_bytes, enc_device_pub_key, iv])

        return self.generate_shared_key_host(priv_key, host_pub_key_bytes, device_id, host_device_id)

    def host_protocol(self, grouped_events, passwords, conn):
        """The host protocol with device identifiers and session identifiers."""
        device_id = "host_device" 
        priv_keys, pub_keys = self.generate_key_pairs(grouped_events)

        final_random_values = []
        for i in range(len(grouped_events)):
            # Step 2: Encrypt and send public keys
            self.send_encrypted_public_key(conn, device_id, i, pub_keys[i], passwords[i])
            # Step 3-7: Receive device's public keys, decrypt them, and exchange random values
            shared_key = self.handle_device_public_key_exchange(conn, priv_keys[i], passwords[i], device_id)
            random_value = self.generate_and_send_random_value(conn, shared_key, i)
            final_random_values.append(random_value)

            # Step 7: Receive and decrypt device's random value
            received_random_value = self.receive_and_decrypt_random_value(conn, shared_key)
            final_random_values.append(received_random_value)

        # Step 8-9: Compute the final key
        return self.compute_final_key(final_random_values)


    def device_protocol(self, grouped_events, passwords, conn, device_id):
        """The device protocol with device identifiers and session identifiers."""
        priv_keys, pub_keys = self.generate_key_pairs(grouped_events)
    
        # Step 2-7: Receive host's public keys, decrypt them, and exchange random values
        final_random_values = []
        for i in range(len(grouped_events)):
            shared_key = self.handle_host_public_key_exchange(conn, priv_keys[i], pub_keys[i], passwords[i], device_id, i)
            received_random_value = self.receive_and_decrypt_random_value(conn, shared_key)
            final_random_values.append(received_random_value)

            # Step 6: Generate and send encrypted random value
            random_value = self.generate_and_send_random_value(conn, shared_key, i)
            final_random_values.append(random_value)

        # Step 8-9: Compute the final key
        return self.compute_final_key(final_random_values)
    
class NotAVerificationKeyException(Exception):
    pass