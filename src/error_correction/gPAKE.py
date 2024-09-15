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

    def generate_shared_key(self, priv_key, peer_pub_key_bytes, local_device_id, peer_device_id):
        """Generate shared key from private key and peer's public key, including device IDs and public keys in the derivation."""
        peer_pub_key = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, peer_pub_key_bytes)
        shared_key = priv_key.exchange(ec.ECDH(), peer_pub_key)

        info = (local_device_id.encode() + peer_device_id.encode() +
            priv_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint) +
            peer_pub_key_bytes)

        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        ).derive(shared_key)

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

    def host_protocol(self, grouped_events, passwords, conn):
        """The host protocol with device identifiers and session identifiers."""
    
        device_id = "host_device"  # Unique identifier for the host device (static)
    
        # Step 1: Generate private/public key pairs for each event type (grouped event)
        priv_keys = [self.generate_priv_key() for _ in grouped_events]
        pub_keys = [self.generate_pub_key(priv_key) for priv_key in priv_keys]
    
        # Step 2: Encrypt each public key with its corresponding password and send it along with device identifier
        for i, (pub_key, pw) in enumerate(zip(pub_keys, passwords)):
            iv = os.urandom(16)  # Generate random IV for encryption
            assert len(iv) == 16, "IV is not 16 bytes before sending"
            print(f"Host: IV before sending (length {len(iv)}): {iv.hex()}")

            symm_key = self.hash_function(pw)  # Derive symmetric key from password
            enc_pub_key = self.encode(symm_key, pub_key, iv)  # Encrypt public key

            enc_pub_key_b64 = base64.b64encode(enc_pub_key).decode()
        
            # Construct session identifier (initially just with host's info, will be completed with device's info later)
            session_id = {'session_idx': i, 'host_device_id': device_id, 'host_enc_pub_key': enc_pub_key_b64}

            device_id_bytes = device_id.encode()  # Convert string to bytes
            session_id_bytes = json.dumps(session_id).encode()  # Serialize session_id to bytes
        
            # Send device identifier, encrypted public key, and IV to the device
            send_pake_msg(conn, [device_id_bytes, session_id_bytes, enc_pub_key, iv])
            print(f"Host: Sent IV (length {len(iv)}): {iv.hex()}")
    
        # Step 3: Receive device's encrypted public keys, session IDs, and decrypt them
        final_random_values = []

        for i in range(len(grouped_events)):
            msg = pake_msg_standby(conn, self.timeout)
            device_id_received = msg[0].decode()  # Receive the device identifier
            session_id_received = json.loads(msg[1].decode())  # Receive the session identifier with both device's and host's public keys
            enc_device_pub_key = msg[2]
            iv_received = msg[3]

            print(f"Host: Received IV (length {len(iv_received)}): {iv_received.hex()}")

            assert len(iv_received) == 16, f"IV is not 16 bytes after receiving, got {len(iv_received)} bytes"

        
            symm_key = self.hash_function(passwords[i])  # Derive decryption key from password
            device_pub_key_bytes = self.decode(symm_key, enc_device_pub_key, iv_received)  # Decrypt device's public key
        
            # Step 4: Generate shared key using ECDH
            shared_key = self.generate_shared_key(priv_keys[i], device_pub_key_bytes, device_id, device_id_received)
        
            # Step 5: Generate random value and encrypt it with the shared key
            random_value = os.urandom(16)  # Generate a 16-byte random value
            iv_random = os.urandom(16)  # Generate IV for random value encryption
            print(f"Host: Generated random IV (length {len(iv_random)}): {iv_random.hex()}")
            encrypted_random_value = self.encode(shared_key, random_value, iv_random)
            session_id_bytes = json.dumps(session_id_received).encode() 
            dummy_value = b''

            assert len(iv_random) == 16, "Random IV is not 16 bytes before sending"
            

            # Step 6: Send the encrypted random value along with the session ID and IV
            send_pake_msg(conn, [session_id_bytes, encrypted_random_value, dummy_value, iv_random])
            print(f"Host: Sent random IV (length {len(iv_random)}): {iv_random.hex()}")

            # Store own random value for summing later
            final_random_values.append(random_value)

        # Step 7: Receive and decrypt the device's random value
        for i in range(len(grouped_events)):
            msg = pake_msg_standby(conn, self.timeout)
            session_id_ver = msg[0]  # Receive the session ID for verification purposes
            enc_random_value_received = msg[1]
            iv_received = msg[2]

            print(f"Host: Received random IV (length {len(iv_received)}): {iv_received.hex()}")

            assert len(iv_received) == 16, "IV is not 16 bytes after receiving in step 7"

            # Decrypt the received random value
            decrypted_random_value = self.decode(shared_key, enc_random_value_received, iv_received)

            # Store decrypted random value for summing
            final_random_values.append(decrypted_random_value)

        # Step 8: Sum all the random values to derive the final group key
        final_group_key = b''.join(final_random_values)

        # Step 9: Hash the final group key for security and output it
        final_key = self.hash_function(final_group_key)
    
        return final_key


    def device_protocol(self, grouped_events, passwords, conn, device_id):
        """The device protocol with device identifiers and session identifiers."""
    
        # Step 1: Generate private/public key pairs for each event type (grouped event)
        priv_keys = [self.generate_priv_key() for _ in grouped_events]
        pub_keys = [self.generate_pub_key(priv_key) for priv_key in priv_keys]
    
        # Step 2: Receive host's encrypted public keys and decrypt them
        final_random_values = []

        for i in range(len(grouped_events)):
            msg = pake_msg_standby(conn, self.timeout)
            host_device_id = msg[0].decode()  # Receive the host's device identifier
            session_id_received = json.loads(msg[1].decode())  # Receive session identifier (contains host's public key and ID)
            enc_host_pub_key = msg[2]
            iv_received = msg[3]

            print(f"Device: IV after receiving (length {len(iv_received)}): {iv_received.hex()}")

            assert len(iv_received) == 16, f"IV is not 16 bytes after receiving, got {len(iv_received)} bytes"
        
            symm_key = self.hash_function(passwords[i])  # Derive decryption key from password
            host_pub_key_bytes = self.decode(symm_key, enc_host_pub_key, iv_received)  # Decrypt host's public key
        
            # Step 3: Encrypt device's public key and send to the host
            iv = os.urandom(16)  # Generate random IV for encryption
            print(f"Device: Generated IV before sending (length {len(iv)}): {iv.hex()}")
            assert len(iv) == 16, "IV is not 16 bytes before sending"
            enc_device_pub_key = self.encode(symm_key, pub_keys[i], iv)  # Encrypt device's public key
        
            # Construct session identifier with device's info and complete it with the host's info
            session_id = {
                'session_idx': i,
                'device_id': device_id,
                'host_device_id': host_device_id,
                'device_enc_pub_key': base64.b64encode(enc_device_pub_key).decode(),
                'host_enc_pub_key': session_id_received['host_enc_pub_key']
            }

            # Convert device_id and session_id to bytes
            device_id_bytes = device_id.encode()  # Convert string to bytes
            session_id_bytes = json.dumps(session_id).encode()  # Serialize session_id to bytes
        
            # Send device identifier, session identifier, encrypted public key, and IV to the host
            send_pake_msg(conn, [device_id_bytes, session_id_bytes, enc_device_pub_key, iv])
            print(f"Device: Sent IV (length {len(iv)}): {iv.hex()}")
        
            # Step 4: Generate shared key using ECDH
            shared_key = self.generate_shared_key(priv_keys[i], host_pub_key_bytes, device_id, host_device_id)
        
            # Step 5: Receive and decrypt the host's random value
            msg = pake_msg_standby(conn, self.timeout)
            session_id_ver = msg[0]
            enc_random_value_received = msg[1]
            iv_received = msg[3]

            if len(msg) == 4:
                dummy_value_received = msg[2] 

            print(f"Device: Received random IV (length {len(iv_received)}): {iv_received.hex()}")
            assert len(iv_received) == 16, f"IV is not 16 bytes after receiving, got {len(iv_received)} bytes"

            decrypted_random_value = self.decode(shared_key, enc_random_value_received, iv_received)  # Decrypt random value

            # Store decrypted random value for summing
            final_random_values.append(decrypted_random_value)

            # Step 6: Generate random value and encrypt it with the shared key
            random_value = os.urandom(16)  # Generate a 16-byte random value
            iv_random = os.urandom(16)  # Generate IV for random value encryption
            print(f"Device: Generated random IV before sending (length {len(iv_random)}): {iv_random.hex()}")
            
            encrypted_random_value = self.encode(shared_key, random_value, iv_random)  # Encrypt random value

            assert len(iv) == 16, "IV is not 16 bytes before sending"

            # Step 7: Send the encrypted random value along with the session ID and IV
            send_pake_msg(conn, [session_id_bytes, encrypted_random_value, iv_random])
            print(f"Device: Sent random IV (length {len(iv_random)}): {iv_random.hex()}")

            # Store own random value for summing later
            final_random_values.append(random_value)

        # Step 8: Sum all the random values to derive the final group key
        final_group_key = b''.join(final_random_values)

        # Step 9: Hash the final group key for security and output it
        final_key = self.hash_function(final_group_key)
    
        return final_key
    
class NotAVerificationKeyException(Exception):
    pass