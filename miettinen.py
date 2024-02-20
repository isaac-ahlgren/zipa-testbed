import numpy as np
from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hmac
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat
import os

# UNTESTED CODE

class Miettinen_Protocol():
    def __init__(self, key_length, n, k, f, w, rel_thresh, abs_thresh, auth_threshold, success_threshold, max_iterations, nfs_server_dir, identifier, timeout):
        self.n = n
        self.k = k
        self.f = f
        self.w = w
        self.rel_thresh = rel_thresh
        self.abs_thresh = abs_thresh
        self.auth_threshold = auth_threshold
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations

        self.timeout = timeout

        self.name = "miettinen"

        self.key_length = key_length
        self.re = Fuzzy_Commitment(ReedSolomonObj(n ,k), key_length)
        self.hash_func = hashes.SHA256()
        self.ec_curve = ec.SECP384R1()
        self.nonce_byte_size = 16

        self.debug = False

        self.count = 0
        self.start = 1 # variable for testing 

    def signal_preprocessing(self, signal, no_snap_shot_width, snap_shot_width):
        block_num = int(len(signal)/(no_snap_shot_width + snap_shot_width))
        c = np.zeros(block_num)
        for i in range(block_num):
            c[i] = np.mean(signal[i*(no_snap_shot_width + snap_shot_width):(i+1)*snap_shot_width])
        return c

    def gen_key(self, c, rel_thresh, abs_thresh):
        bits = ""
        for i in range(len(c)-1):
            feature1 = np.abs(c[i]/(c[i-1]) - 1)
            feature2 = np.abs(c[i] - c[i-1])
            if feature1 > rel_thresh and feature2 > abs_thresh:
                bits += '1'
            else:
                bits += '0'
        return bits

    def miettinen_algo(self, x):
        import random
        #def bitstring_to_bytes(s):
        #    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
        #signal = self.signal_preprocessing(x, self.f, self.w)
        #key = self.gen_key(signal, self.rel_thresh, self.abs_thresh)
        #return bitstring_to_bytes(key)

        b = bytearray([i for i in range(self.start, self.start+self.n)])
        if self.debug:
            b[0] = 0
        self.start += self.n
        return b

    def extract_context(self):
        print()
        print("Extracting Context")
        #signal = self.signal_measurement.get_audio()
        signal = None
        bits = self.miettinen_algo(signal)
        print()
        return bits, signal

    def device_protocol(self, host_socket):
        self.debug = True
        host_socket.setblocking(1)
        print("Iteration " + str(self.count))

        # Sending ack that they are ready to begin
        print()
        print("Sending ACK")
        ack(host_socket)

        # Wait for ack from host to being context extract, quit early if no response within time
        print()
        print("Waiting for ACK from host")
        if not ack_standby(host_socket, self.timeout):
            print("No ACK recieved within time limit - early exit")
            print()
            return
        print()

        # Generate initial private key for Diffie-Helman
        initial_private_key = ec.generate_private_key(ec.SECP384R1())
        
        public_key = initial_private_key.public_key().public_bytes(Encoding.X962, PublicFormat.CompressedPoint)
        
        # Send initial key for Diffie-Helman
        print("Send DH public key\n")
        dh_exchange(host_socket, public_key)

        # Recieve other devices key
        print("Waiting for DH public key\n")
        other_public_key_bytes = dh_exchange_standby(host_socket, self.timeout)

        if other_public_key_bytes == None:
            print("No initial key for Diffie-Helman recieved - early exit")
            print()
            return

        other_public_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP384R1(), other_public_key_bytes)

        # Shared key generated
        shared_key = initial_private_key.exchange(ec.ECDH(), other_public_key)

        current_key = shared_key
        successes = 0
        total_iterations = 0
        while successes < self.success_threshold and total_iterations < self.max_iterations:
            # Sending ack that they are ready to begin

            print("Waiting for ACK from host.")
            if not ack_standby(host_socket, self.timeout):
                print("No ACK recieved within time limit - early exit.\n\n")
                return
            
            print()
            print("Sending ACK")
            ack(host_socket)

            success = False

            # Extract bits from mic
            witness, signal = self.extract_context()
        
            # Wait for Commitment
            print("Waiting for commitment from host")
            commitment, h = commit_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not commitment:
                print("No commitment recieved within time limit - early exit\n")
                return
            print()

            print("witness: " + str(witness))
            print()

            # Decommit
            print("Decommiting")
            prederived_key = self.re.decommit_witness(commitment, witness)

            kdf = HKDF(algorithm=hashes.SHA256(), length=self.key_length, salt=None, info=None)
            derived_key = kdf.derive(prederived_key + current_key)

            # Key Confirmation Phase

            # Hash prederived key
            hash_func = hashes.Hash(hashes.SHA512())
            hash_func.update(prederived_key)
            pd_key_hash = hash_func.finalize()

            # Generate Nonce
            nonce = os.urandom(16)

            # Create tag of Nonce
            mac = hmac.HMAC(derived_key, hashes.SHA256())
            mac.update(nonce)
            tag = mac.finalize()

            # Create key confirmation message
            hash = pd_key_hash + nonce + tag

            send_hash(host_socket, hash)

            host_hash = get_nonce_standby(host_socket, self.timeout)        

            # Early exist if no commitment recieved in time
            if not host_hash:
                print("No hash recieved within time limit - early exit")
                print()
                return
            print()

            # If hashes are equal, then it was successful
            if constant_time.bytes_eq(host_hash, hash):
                success = True
                successes += 1
                current_key = derived_key

            print("Produced Key: " + str(derived_key))
            print("success: " + str(success))
            print()

            # Log all information to NFS server
            print("Logging all information to NFS server")
            
            # Lets not log anything yet
            #self.send_to_nfs_server("audio", signal, witness, h, commitment)

            # Increment total number of iterations key evolution has occured
            total_iterations += 1
        
        if successes/total_iterations >= self.auth_threshold:
            print("Total Key Pairing Success: auth - " + str(successes/total_iterations))
        else:
            print("Total Key Pairing Failure: auth - " + str(successes/total_iterations))

        self.count += 1

    def host_protocol(self, device_sockets):
        print("Iteration " + str(self.count))
        print()
  
        participating_sockets = ack_all_standby(device_sockets, self.timeout)

        # Exit early if no devices to pair with
        if len(participating_sockets) == 0:
            print("No advertised devices joined the protocol - early exit")
            print()
            return
        print("Successfully ACKed participating devices")
        print()

        print("ACKing all participating devices")
        ack_all(participating_sockets)

        # Generate initial private key for Diffie-Helman
        initial_private_key = ec.generate_private_key(ec.SECP384R1())
        
        # Obtain Public Key
        public_key = initial_private_key.public_key().public_bytes(Encoding.X962, PublicFormat.CompressedPoint)

        # Send initial key for Diffie-Helman
        print("Send every device the public key\n")
        dh_exchange_all(participating_sockets, public_key)

        # Recieve other devices key
        print('Recieve every devices public key\n')
        participating_sockets, keys_recieved = dh_exchange_standby_all(participating_sockets, self.timeout)

        if len(participating_sockets) == 0:
            print("No advertised devices joined the protocol - early exit")
            print()
            return

        # Generating Shared Keys
        successes = dict()
        for i in range(len(participating_sockets)):
            ipaddr, port = participating_sockets[i].getpeername()
            public_key_bytes = keys_recieved[ipaddr]
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP384R1(), public_key_bytes)
            keys_recieved[ipaddr] = initial_private_key.exchange(ec.ECDH(), public_key)
            successes[ipaddr] = 0

        done_pairing = []
        current_keys = keys_recieved
        total_iterations = 0
        while len(participating_sockets) != 0 and total_iterations < self.max_iterations:
            # ACK all devices
            ack_all(participating_sockets)

            participating_sockets = ack_all_standby(participating_sockets, self.timeout)

            # Exit early if no devices to pair with
            if len(participating_sockets) == 0:
                print("No advertised devices joined the protocol - early exit")
                print()
                return
            print("Successfully ACKed participating devices")
            print()

            # Extract key from mic
            witness, signal = self.extract_context()

            # Commit Secret
            print("Commiting Witness")
            prederived_key, commitment = self.re.commit_witness(witness)

            print("witness: " + str(witness))
            print()

            print("Sending commitment")
            print()
            h = bytes([0 for i in range(64)]) # Scheme does not send hash but function expects 64 byte hash (this is gonna change in the future)
            send_commit(commitment, h, participating_sockets)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Recieve all nonces
            participating_sockets, recieved_nonces = send_nonce_standby_all(participating_sockets, self.timeout)
            derived_keys = dict() # dictionary to keep track of derived keys
            for i in range(len(participating_sockets)):
                # Derive new key using previous key and new prederived key from fuzzy commitment
                ipaddr, port = participating_sockets[i].getpeername()
                kdf = HKDF(algorithm=self.hash_func, length=self.key_length, salt=None, info=None)
                derived_key = kdf.derive(prederived_key + current_key)
                derived_keys[ipaddr] = derived_key

                # Retrieve nonce for device
                recieved_nonce = recieved_nonces[ipaddr]

                if self.verify_mac_from_device(recieved_nonce, derived_key, pd_key_hash):
                    successes[ipaddr] += 1
                    current_keys[ipaddr] = derived_key

            derived_keys = dict() # dictionary to keep track of derived keys
            for i in range(len(participating_sockets)):
                # Get current sockets current key
                ipaddr, port = participating_sockets[i].getpeername()
                current_key = current_keys[ipaddr]

                # Derive new key using previous key and new prederived key from fuzzy commitment
                kdf = HKDF(algorithm=self.hash_func, length=self.key_length, salt=None, info=None)
                derived_key = kdf.derive(prederived_key + current_key)
                derived_keys[ipaddr] = derived_key

                # Send key confirmation value
                self.send_key_confirmation(participating_sockets[i], pd_key_hash, derived_key)

            # Increment total times key evolution has occured
            total_iterations += 1

        self.count += 1

    def hash_function(self, bytes):
        hash_func = hashes.Hash(hashes.SHA512())
        hash_func.update(bytes)
        return hash_func.finalize()

    def send_nonce_msg_to_device(self, connection, recieved_nonce, derived_key):
        nonce = os.urandom(self.nonce_byte_size)

        # Concatenate nonces together
        concat_nonce = nonce + recieved_nonce

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(concat_nonce)
        tag = mac.finalize()

        # Construct nonce message
        nonce_msg = nonce + tag

        send_nonce(connection, nonce_msg)

    def send_nonce_msg_to_host(self, connection, prederived_key_hash, derived_key):
        # Generate Nonce
        nonce = os.urandom(self.nonce_byte_size)

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(nonce)
        tag = mac.finalize()

        # Create key confirmation message
        nonce_msg = prederived_key_hash + nonce + tag
                 
        send_nonce(connection, nonce_msg)

    def verify_mac_from_host(self, recieved_nonce_msg, generated_nonce, derived_key):
        recieved_nonce = recieved_nonce_msg[0:self.nonce_byte_size]

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce + generated_nonce)
        generated_tag = mac.finalize()
        
        recieved_tag = recieved_nonce_msg[self.nonce_byte_size:]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success

    def verify_mac_from_device(self, recieved_nonce_msg, derived_key, prederived_key_hash):
        success = False
        
        # Retrieve nonce used by device
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[pd_hash_len:pd_hash_len + self.nonce_byte_size]

        # Generate new MAC tag for the nonce with respect to the derived key
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[pd_hash_len + self.nonce_byte_size:]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success

import socket
def device(prot):
    print("device")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))
    prot.device_protocol(s)

def host(prot):
    print("host")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(0)
    prot.host_protocol([conn])

if __name__ == "__main__":
    import multiprocessing as mp
    import random
    random.seed(0)
    prot = Miettinen_Protocol(8, 12, 8, 5*48000, 6*48000, 0.5, 0.5, 0.9, 5, 20, "", 0, 30)
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()