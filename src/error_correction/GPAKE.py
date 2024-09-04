import hashlib
import os
import secrets
import traceback  # Import to help with detailed error tracing

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
        #Step 3
        self.curve = ec.SECP256R1()  # Example elliptic curve choice
        # Generate a private key and use its public key as the generator point
        temp_private_key = ec.generate_private_key(self.curve)
        self.P = temp_private_key.public_key().public_numbers().x  # Use the x-coordinate as an example generator value
        self.hash_algo = hashes.BLAKE2b(64)  # BLAKE2b with a 64-byte digest
        self.encryption_algo = ChaCha20Poly1305  # Authenticated encryption with ChaCha20-Poly1305
        self.key_length = key_length
        self.timeout = timeout

        # Generate elliptic curve parameters
        self.private_key = ec.generate_private_key(self.curve)
        self.public_key = self.private_key.public_key()

    # Other initialization methods if necessary
    #Step 4
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

    #Step 5
    def encrypt_public_keys(self, event_keys, passwords):
        encrypted_keys = {}
        for event_type, keys in event_keys.items():
            pub_key_bytes = keys['public_key'].public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint
            )
            # Derive a key from the password
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'GPAKE encryption',
            )
            derived_key = hkdf.derive(passwords[event_type])
            #password = passwords[event_type] #.encode('utf-8')
            cipher = ChaCha20Poly1305(derived_key)  # Using a 32-byte key for ChaCha20-Poly1305
            nonce = secrets.token_bytes(12)  # ChaCha20-Poly1305 requires a 12-byte nonce
            encrypted_pub_key = cipher.encrypt(nonce, pub_key_bytes, None)
            encrypted_keys[event_type] = (nonce, encrypted_pub_key)
        return encrypted_keys
    
    #Step 6
    def broadcast_encrypted_keys(self, conn, encrypted_keys):
        for event_type, (nonce, encrypted_key) in encrypted_keys.items():
            send_gpake_msg(conn, [nonce, encrypted_key])

    #Step 7 and 8
    def receive_and_decrypt_keys(self, conn, passwords):
        valid_keys = {}
        print("Device: receive_and_decrypt_keys started", flush=True)
        
        used_passwords = set()

        num_expected_keys = len(passwords)
        num_received_keys = 0
        
        while num_received_keys < num_expected_keys:
            print("Device: Waiting for data...", flush=True)
            received_data = gpake_msg_standby(conn, self.timeout)
            if received_data is None:
                print("Device: No more data to receive or timed out", flush=True)
                break  # No more data to receive or timed out
        
            print(f"Device: Received data: {received_data}", flush=True)
            nonce, encrypted_pub_key = received_data

            for password in passwords:
                if password in used_passwords:
                    # Skip already used passwords
                    print(f"Device: Skipping password {password} as it has already been successful", flush=True)
                    continue

                try:
                    print(f"Device: Trying to decrypt with password: {password}", flush=True)
                    hkdf = HKDF(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=None,
                        info=b'GPAKE encryption',
                    )
                    derived_key = hkdf.derive(password)
                    cipher = ChaCha20Poly1305(derived_key)
                    pub_key_bytes = cipher.decrypt(nonce, encrypted_pub_key, None)
                    print(f"Device: Successfully decrypted bytes: {pub_key_bytes}", flush=True)
                
                    # Deserialize the public key
                    pub_key = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, pub_key_bytes)
                    print(f"Device: Successfully decrypted and deserialized public key: {pub_key}", flush=True)
                
                    # Check if the public key is valid on the curve
                    if isinstance(pub_key, ec.EllipticCurvePublicKey):
                        valid_keys[password] = pub_key
                        used_passwords.add(password)  # Mark password as used
                        num_received_keys += 1  # Increment the number of received keys
                        print(f"Device: Valid public key found with password: {password}", flush=True)
                        break  # Exit the loop once a valid key is found
                except Exception as e:
                    print(f"Device: Decryption failed with password: {password}", flush=True)
                    traceback.print_exc()  # Detailed error output
                    # If decryption or validation fails, continue with the next password
                    continue

            # Check if all keys have been processed
            if num_received_keys >= num_expected_keys:
                print(f"Device: All {num_received_keys} keys successfully received and decrypted.", flush=True)
                break

        return valid_keys

    #Step 9
    def generate_session_ids(self, local_id, remote_ids, valid_keys):
        session_ids = {}
        for password, pub_key in valid_keys.items():
            for remote_id in remote_ids:
                session_id = f"{local_id}-{remote_id}-{password}"
                session_ids[password] = session_id
        return session_ids

    #Step 10
    def derive_ecdh_keys(self, private_key, valid_keys):
        ecdh_keys = {}
        for password, pub_key in valid_keys.items():
            shared_key = private_key.exchange(ec.ECDH(), pub_key)
        
            # Optionally, you can hash the shared key to derive the final key material
            derived_key = hashlib.blake2b(shared_key).digest()
            ecdh_keys[password] = derived_key
            print(f"Derived ECDH key for event type {event_type}: {derived_key.hex()}", flush=True)
        return ecdh_keys
    
    #Step 11
    def generate_random_values(self, grouped_events):
        random_values = {}
        for group_index, events in enumerate(grouped_events):
            random_value = secrets.token_bytes(self.key_length)
            random_values[group_index] = random_value
            print(f"Generated random value for group {group_index}: {random_value.hex()}", flush=True)
        return random_values
    
    #Step 11.5
    def encrypt_random_values(self, random_values, ecdh_keys):
        encrypted_values = {}
        for event_type, random_value in random_values.items():
            if event_type not in ecdh_keys:
                print(f"Error: No ECDH key found for event type {event_type}", flush=True)
                print(f"Available ECDH keys: {list(ecdh_keys.keys())}", flush=True)
                raise KeyError(f"Missing ECDH key for event type {event_type}")
            key = ecdh_keys[event_type]
            cipher = ChaCha20Poly1305(key)
            nonce = secrets.token_bytes(12)  # ChaCha20-Poly1305 requires a 12-byte nonce
            encrypted_value = cipher.encrypt(nonce, random_value, None)
            encrypted_values[event_type] = (nonce, encrypted_value)
        return encrypted_values
    
    #Step 12
    def broadcast_encrypted_values(self, conn, local_id, session_ids, encrypted_values):
        for event_type, (nonce, encrypted_value) in encrypted_values.items():
            session_id = session_ids[event_type]
            message = [local_id.encode('utf-8'), session_id.encode('utf-8'), nonce, encrypted_value]
            send_gpake_msg(conn, message)

    #Step 13
    def check_session_id(self, session_id, local_id):
        return local_id in session_id
    
    #Step 14
    def decrypt_random_value(self, encrypted_value, nonce, intermediate_key):
        cipher = ChaCha20Poly1305(intermediate_key)
        random_value = cipher.decrypt(nonce, encrypted_value, None)
        return random_value
    
    #Step 15
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

    def host_protocol(self, passwords, grouped_events, conn):
        print("Host protocol started")
        # Step 1: Generate key pairs for each event group
        event_keys = self.generate_key_pairs_for_events(grouped_events)
        print("Host: Generated key pairs")
        # Step 2: Encrypt public keys using passwords
        encrypted_keys = self.encrypt_public_keys(event_keys, passwords)
        print("Host: Encrypted public keys")
        # Step 3: Broadcast encrypted public keys
        self.broadcast_encrypted_keys(conn, encrypted_keys)
        print("Host: Broadcasted encrypted public keys")
        # Step 4: Receive and decrypt public keys from other devices
        valid_keys = self.receive_and_decrypt_keys(conn, passwords)
        print("Host: Received and decrypted public keys")
        # Step 5: Generate session IDs using valid public keys
        session_ids = self.generate_session_ids("host", ["device"], valid_keys)
        
        # Step 6: Derive intermediate ECDH keys
        private_key = self.private_key  # Host uses its existing private key
        ecdh_keys = self.derive_ecdh_keys(private_key, valid_keys)
        print("Host: Derived intermediate ECDH keys")
        
        # Step 7: Generate random values for each event group
        random_values = self.generate_random_values(grouped_events)
    
        # Step 8: Encrypt the random values using the intermediate ECDH keys
        encrypted_values = self.encrypt_random_values(random_values, ecdh_keys)
        print("Host: Encrypted random values")
    
        # Step 9: Broadcast the encrypted random values along with session IDs
        self.broadcast_encrypted_values(conn, "host", session_ids, encrypted_values)
        print("Host: Broadcasted encrypted random values")
    
        # Step 10: Process incoming messages to retrieve the random values sent by other devices
        derived_random_values = self.process_incoming_messages(conn, "host", ecdh_keys)
        print("Host: Processed incoming messages")
    
        # Step 11: Derive the group key by summing or using a KDF on the collected random values
        group_key = self.derive_group_key(derived_random_values)
    
        # Step 12: Notify other devices about the success of key establishment
        send_status(conn, True)

        return group_key


    def device_protocol(self, passwords, grouped_events, conn, local_id, remote_ids):
        print("Device protocol started")
        # Step 1: Receive and decrypt the encrypted public keys using all passwords
        valid_keys = self.receive_and_decrypt_keys(conn, passwords)
        print("Device: Received and decrypted public keys")
    
        # Step 2: Generate session IDs for the valid keys
        session_ids = self.generate_session_ids(local_id, remote_ids, valid_keys)
        print("Device: Generated session IDs")
    
        # Step 3: Derive the intermediate ECDH keys using the private key and the valid public keys
        private_key = ec.generate_private_key(self.curve)  # Assume each device has its private key
        ecdh_keys = self.derive_ecdh_keys(private_key, valid_keys)
        print("Device: Derived intermediate ECDH keys")
    
        # Step 4: Generate random values for each grouped event
        random_values = self.generate_random_values(grouped_events)
    
        # Step 5: Encrypt the random values using the intermediate ECDH keys
        encrypted_values = self.encrypt_random_values(random_values, ecdh_keys)
        print("Device: Encrypted random values")
    
        # Step 6: Broadcast the encrypted random values along with the session ID and device ID
        self.broadcast_encrypted_values(conn, local_id, session_ids, encrypted_values)
        print("Device: Broadcasted encrypted random values")
    
        # Step 7: Process incoming messages to retrieve the random values sent by other devices
        derived_random_values = self.process_incoming_messages(conn, local_id, ecdh_keys)
        print("Device: Processed incoming messages")
    
        # Step 8: Derive the group key by summing or using a KDF on the collected random values
        group_key = self.derive_group_key(derived_random_values)
        print("Device: Derived group key")
        print("Device protocol finished")
    
        return group_key

