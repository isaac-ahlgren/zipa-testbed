import hashlib
import os
import secrets

import zmq
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from networking.network import (
    gpake_msg_standby,
    send_gpake_msg,
    send_status,
    status_standby,
)


class PartitionedGPAKE:
    def __init__(self, key_length, timeout):
        # Initialize cryptographic components
        self.curve = ec.SECP256R1()  # Example elliptic curve choice
        self.P = self.curve.generator
        self.hash_algo = hashes.BLAKE2b(64)  # BLAKE2b with a 64-byte digest
        self.encryption_algo = (
            ChaCha20Poly1305  # Authenticated encryption with ChaCha20-Poly1305
        )
        self.key_length = key_length
        self.timeout = timeout

        # Generate elliptic curve parameters
        self.private_key = ec.generate_private_key(self.curve)
        self.public_key = self.private_key.public_key()

    # Other initialization methods if necessary

    def generate_key_pair(self):
        priv_key = ec.generate_private_key(self.curve)
        pub_key = priv_key.public_key()
        return priv_key, pub_key
    
    def generate_key_pairs_for_events(self, grouped_events):
        event_keys = {}
        for event_type, events in enumerate(grouped_events):
            priv_key, pub_key = self.generate_key_pair()
            event_keys[event_type] = {
                'private_key': priv_key,
                'public_key': pub_key
            }
        return event_keys

    
    def encrypt_public_keys(self, event_keys, passwords):
        encrypted_keys = {}
        for event_type, keys in event_keys.items():
            pub_key_bytes = keys['public_key'].public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint
            )
            # Derive a key from the password
            password = passwords[event_type].encode('utf-8')
            cipher = ChaCha20Poly1305(password[:32])  # Using a 32-byte key for ChaCha20-Poly1305
            nonce = secrets.token_bytes(12)  # ChaCha20-Poly1305 requires a 12-byte nonce
            encrypted_pub_key = cipher.encrypt(nonce, pub_key_bytes, None)
            encrypted_keys[event_type] = (nonce, encrypted_pub_key)
        return encrypted_keys
    
    def broadcast_encrypted_keys(self, conn, encrypted_keys):
        for event_type, (nonce, encrypted_key) in encrypted_keys.items():
            send_gpake_msg(conn, [nonce, encrypted_key])

    def receive_and_decrypt_keys(self, conn, passwords):
        valid_keys = {}
        while True:
            received_data = gpake_msg_standby(conn, self.timeout)
            if received_data is None:
                break  # No more data to receive or timed out
        
            nonce, encrypted_pub_key = received_data
            for password in passwords:
                try:
                    cipher = ChaCha20Poly1305(password.encode('utf-8')[:32])
                    pub_key_bytes = cipher.decrypt(nonce, encrypted_pub_key, None)
                
                    # Deserialize the public key
                    pub_key = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, pub_key_bytes)
                
                    # Check if the public key is valid on the curve
                    if isinstance(pub_key, ec.EllipticCurvePublicKey):
                        valid_keys[password] = pub_key
                        break  # Exit the loop once a valid key is found
                except Exception as e:
                    # If decryption or validation fails, continue with the next password
                    continue
        return valid_keys

    def generate_session_ids(self, local_id, remote_ids, valid_keys):
        session_ids = {}
        for password, pub_key in valid_keys.items():
            for remote_id in remote_ids:
                session_id = f"{local_id}-{remote_id}-{password}"
                session_ids[password] = session_id
        return session_ids

    def derive_ecdh_keys(self, private_key, valid_keys):
        ecdh_keys = {}
        for password, pub_key in valid_keys.items():
            shared_key = private_key.exchange(ec.ECDH(), pub_key)
        
            # Optionally, you can hash the shared key to derive the final key material
            derived_key = hashlib.blake2b(shared_key).digest()
            ecdh_keys[password] = derived_key
        return ecdh_keys
    
    def generate_random_values(self, grouped_events):
        random_values = {}
        for group_index, events in enumerate(grouped_events):
            random_value = secrets.token_bytes(self.key_length)
            random_values[group_index] = random_value
        return random_values
    
    def encrypt_random_values(self, random_values, ecdh_keys):
        encrypted_values = {}
        for event_type, random_value in random_values.items():
            key = ecdh_keys[event_type]
            cipher = ChaCha20Poly1305(key)
            nonce = secrets.token_bytes(12)  # ChaCha20-Poly1305 requires a 12-byte nonce
            encrypted_value = cipher.encrypt(nonce, random_value, None)
            encrypted_values[event_type] = (nonce, encrypted_value)
        return encrypted_values
    
    def broadcast_encrypted_values(self, conn, local_id, session_ids, encrypted_values):
        for event_type, (nonce, encrypted_value) in encrypted_values.items():
            session_id = session_ids[event_type]
            message = [local_id.encode('utf-8'), session_id.encode('utf-8'), nonce, encrypted_value]
            send_gpake_msg(conn, message)

    def check_session_id(self, session_id, local_id):
        return local_id in session_id
    
    def decrypt_random_value(self, encrypted_value, nonce, intermediate_key):
        cipher = ChaCha20Poly1305(intermediate_key)
        random_value = cipher.decrypt(nonce, encrypted_value, None)
        return random_value
    
    def process_incoming_messages(self, conn, local_id, ecdh_keys):
        derived_random_values = {}
    
        while True:
            received_data = gpake_msg_standby(conn, self.timeout)
            if received_data is None:
                break  # No more data to receive or timed out
        
            # Unpack the received message
            remote_id, session_id, nonce, encrypted_value = received_data
            session_id = session_id.decode('utf-8')
        
            # Check if the local device ID is in the session ID
            if self.check_session_id(session_id, local_id):
                # Find the correct intermediate key based on the session ID
                for group_index, key in ecdh_keys.items():
                    if session_id in key:  # Assuming session_id is a part of the key
                        intermediate_key = ecdh_keys[group_index]
                        break
            
                # Decrypt the random value
                random_value = self.decrypt_random_value(encrypted_value, nonce, intermediate_key)
            
                # Store the derived random value with the corresponding session ID
                derived_random_values[session_id] = random_value
    
        return derived_random_values
    
    def derive_group_key(self, derived_random_values):
        combined_key_material = b''
        for session_id, random_value in derived_random_values.items():
            combined_key_material += random_value
    
        # If you are confident in the randomness of the collected values
        group_key = hashlib.blake2b(combined_key_material).digest()
        return group_key
    
    def derive_group_key_with_kdf(self, derived_random_values):
        combined_key_material = b''
        for session_id, random_value in derived_random_values.items():
            combined_key_material += random_value
    
        # Use HKDF to derive a more secure key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=None,
            info=b'GPAKE group key derivation',
        )
        group_key = hkdf.derive(combined_key_material)
        return group_key





    





    def decrypt_public_key(self, encrypted_pub_key, password):
        pub_key = self.encryption_algo.decrypt(encrypted_pub_key, password)
        return pub_key

    def host_protocol(self, passwords, conn, signal_data):
        inter_event_timings, grouped_events = IoTCupidProcessing.iotcupid(signal_data, ...)
    
        event_keys = self.generate_key_pairs_for_events(grouped_events)
        encrypted_keys = self.encrypt_public_keys(event_keys, passwords)
    
        self.broadcast_encrypted_keys(conn, encrypted_keys)



        received_data = gpake_msg_standby(conn, self.timeout)
        if received_data is None:
            send_status(conn, False)
            return None

        decrypted_pub_key = self.decrypt_public_key(received_data[0], password)

        session_id = self.derive_session_id(pub_key, decrypted_pub_key)
        intermediate_key = self.derive_intermediate_key(priv_key, decrypted_pub_key)

        session_keys.append(intermediate_key)

        group_key = self.derive_group_key(session_keys)
        send_status(conn, True)
        return group_key

    def device_protocol(self, passwords, conn, local_id, remote_ids):
        # 1. Process the signal data using IoTCupid
        inter_event_timings, grouped_events = IoTCupidProcessing.iotcupid(signal_data, ...)
    
        # 2. Receive and decrypt the encrypted public keys using all passwords
        valid_keys = self.receive_and_decrypt_keys(conn, passwords)
    
        # 3. Generate session IDs for the valid keys
        session_ids = self.generate_session_ids(local_id, remote_ids, valid_keys)
    
        # 4. Derive the intermediate ECDH keys using the private key and the valid public keys
        private_key = ec.generate_private_key(self.curve)  # Assume each device has its private key
        ecdh_keys = self.derive_ecdh_keys(private_key, valid_keys)
    
        # 5. Generate random values for each grouped event
        random_values = self.generate_random_values(grouped_events)
    
        # 6. Encrypt the random values using the intermediate ECDH keys
        encrypted_values = self.encrypt_random_values(random_values, ecdh_keys)
    
        # 7. Broadcast the encrypted random values along with the session ID and device ID
        self.broadcast_encrypted_values(conn, local_id, session_ids, encrypted_values)
    
        # 8. Process incoming messages to retrieve the random values sent by other devices
        derived_random_values = self.process_incoming_messages(conn, local_id, ecdh_keys)

        # 9. Derive the group key by summing or using a KDF on the collected random values
        group_key = self.derive_group_key(derived_random_values)
    
        return group_key
