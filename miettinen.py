# TODO compare with Seemoo lab implementation: https://github.com/seemoo-lab/ubicomp19_zero_interaction_security/blob/master/Visualization/Miettinen.ipynb
import multiprocessing as mp
import os

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj


class Miettinen_Protocol:
    def __init__(
        self,
        sensor,
        key_length,
        parity_symbols,
        f,
        w,
        rel_thresh,
        abs_thresh,
        auth_threshold,
        success_threshold,
        max_iterations,
        timeout,
        logger,
        verbose=True,
    ):
        self.sensor = sensor
        self.f = int(f * self.sensor.sensor.sample_rate)
        self.w = int(w * self.sensor.sensor.sample_rate)
        self.rel_thresh = rel_thresh
        self.abs_thresh = abs_thresh
        self.auth_threshold = auth_threshold
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations

        self.timeout = timeout
        self.name = "miettinen"

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, key_length), key_length
        )
        self.hash_func = hashes.SHA256()
        self.ec_curve = ec.SECP384R1()
        self.nonce_byte_size = 16

        self.time_length = (w + f) * (self.commitment_length * 8 + 1)

        self.logger = logger

        self.count = 0

        self.verbose = verbose

    def signal_preprocessing(self, signal, no_snap_shot_width, snap_shot_width):
        block_num = int(len(signal) / (no_snap_shot_width + snap_shot_width))
        c = np.zeros(block_num)
        for i in range(block_num):
            c[i] = np.mean(
                signal[
                    i
                    * (no_snap_shot_width + snap_shot_width) : i
                    * (no_snap_shot_width + snap_shot_width)
                    + snap_shot_width
                ]
            )
        return c

    def gen_key(self, c, rel_thresh, abs_thresh):
        bits = ""
        for i in range(len(c) - 1):
            feature1 = np.abs(c[i] / (c[i - 1]) - 1)
            feature2 = np.abs(c[i] - c[i - 1])
            if feature1 > rel_thresh and feature2 > abs_thresh:
                bits += "1"
            else:
                bits += "0"
        return bits

    # TODO: algorithm needs to be testing using real life data
    def miettinen_algo(self, x):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        signal = self.signal_preprocessing(x, self.f, self.w)
        key = self.gen_key(signal, self.rel_thresh, self.abs_thresh)
        return bitstring_to_bytes(key)

    def extract_context(self):
        signal = self.sensor.read(
            int(self.time_length * self.sensor.sensor.sample_rate)
        )
        bits = self.miettinen_algo(signal)
        return bits, signal

    def parameters(self, is_host):
        parameters = ""
        parameters += "protocol: " + self.name
        parameters += "is_host: " + str(is_host)
        parameters += (
            "\nsensor: " + self.sensor.sensor.name
        )  # Name's in physical sensor.
        parameters += "\nkey_length: " + str(self.key_length)
        parameters += "\nparity_symbols: " + str(self.parity_symbols)
        parameters += "\nf: " + str(self.f)
        parameters += "\nw: " + str(self.w)
        parameters += "\nrel_thresh: " + str(self.rel_thresh)
        parameters += "\nabs_thresh: " + str(self.abs_thresh)
        parameters += "\nsuccess_threshold: " + str(self.success_threshold)
        parameters += "\nauthentication_threshold: " + str(self.auth_threshold)
        parameters += "\nmax_iterations: " + str(self.max_iterations)
        parameters += "\ntime_length: " + str(self.time_length)
        return parameters

    def device_protocol(self, host_socket):
        host_socket.setblocking(1)

        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("\nSending ACK")
        ack(host_socket)

        # Wait for ack from host to being context extract, quit early if no response within time
        if self.verbose:
            print("\nWaiting for ACK from host")

        if not ack_standby(host_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit\n")
            return

        # Shared key generated
        shared_key = self.diffie_hellman(host_socket)

        current_key = shared_key
        successes = 0
        total_iterations = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            # Sending ack that they are ready to begin

            if self.verbose:
                print("Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Sending ACK\n")
            ack(host_socket)

            success = False

            # Extract bits from sensor
            witness, signal = self.extract_context()

            # Wait for Commitment
            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)

            commitment = commitments[0]

            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return

            if self.verbose:
                print("witness: " + str(witness))

            # Decommit

            if self.verbose:
                print("Decommiting")
            prederived_key = self.re.decommit_witness(commitment, witness)

            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Send nonce message to host
            generated_nonce = self.send_nonce_msg_to_host(
                host_socket, pd_key_hash, derived_key
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                return

            # If hashes are equal, then it was successful
            if self.verify_mac_from_host(
                recieved_nonce_msg, generated_nonce, derived_key
            ):
                success = True
                successes += 1
                current_key = derived_key

            if self.verbose:
                print("Produced Key: " + str(derived_key))
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(total_iterations)
                )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=total_iterations,
            )

            # Increment total number of iterations key evolution has occured
            total_iterations += 1

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / total_iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / total_iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(total_iterations)
                    + " succeeded: "
                    + str(successes / total_iterations >= self.auth_threshold),
                )
            ]
        )

        self.count += 1

    def host_protocol(self, device_sockets):
        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count) + "\n")
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket):
        device_socket.setblocking(1)

        device_ip_addr, device_port = device_socket.getpeername()

        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return

        if self.verbose:
            print("ACKing participating device")
        ack(device_socket)

        # Shared key generated
        shared_key = self.diffie_hellman(device_socket)

        current_key = shared_key
        total_iterations = 0
        successes = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            success = False
            # ACK device
            if self.verbose:
                ack(device_socket)

            if not ack_standby(device_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Successfully ACKed participating device\n")

            # Extract key from sensor
            witness, signal = self.extract_context()

            # Commit Secret
            if self.verbose:
                print("Commiting Witness")
            prederived_key, commitment = self.re.commit_witness(witness)

            if self.verbose:
                print("witness: " + str(witness))
                print()

            if self.verbose:
                print("Sending commitment")
                print()

            send_commit([commitment], None, device_socket)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                    print()
                return

            # Derive new key using previous key and new prederived key from fuzzy commitment
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            if self.verify_mac_from_device(
                recieved_nonce_msg, derived_key, pd_key_hash
            ):
                success = True
                successes += 1
                current_key = derived_key

            # Create and send key confirmation value
            self.send_nonce_msg_to_device(
                device_socket, recieved_nonce_msg, derived_key, pd_key_hash
            )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=total_iterations,
                ip_addr=device_ip_addr,
            )

            # Increment total times key evolution has occured
            total_iterations += 1

            if self.verbose:
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(total_iterations)
                )
                print()

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / total_iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / total_iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(total_iterations)
                    + " succeeded: "
                    + str(successes / total_iterations >= self.auth_threshold),
                )
            ],
            ip_addr=device_ip_addr,
        )

        self.count += 1

    # TODO: Refactor by putting in common_protocols.py, then change all references to this version to the common_protocols.py version
    def diffie_hellman(self, socket):
        # Generate initial private key for Diffie-Helman
        initial_private_key = ec.generate_private_key(self.ec_curve)

        public_key = initial_private_key.public_key().public_bytes(
            Encoding.X962, PublicFormat.CompressedPoint
        )

        # Send initial key for Diffie-Helman
        if self.verbose:
            print("Send DH public key\n")

        dh_exchange(socket, public_key)

        # Recieve other devices key
        if self.verbose:
            print("Waiting for DH public key\n")

        other_public_key_bytes = dh_exchange_standby(socket, self.timeout)

        if other_public_key_bytes == None:
            if self.verbose:
                print("No initial key for Diffie-Helman recieved - early exit\n")
            return

        other_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            self.ec_curve, other_public_key_bytes
        )

        # Shared key generated
        shared_key = initial_private_key.exchange(ec.ECDH(), other_public_key)

        return shared_key

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def send_nonce_msg_to_device(
        self, connection, recieved_nonce_msg, derived_key, prederived_key_hash
    ):
        nonce = os.urandom(self.nonce_byte_size)

        # Concatenate nonces together
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len : pd_hash_len + self.nonce_byte_size
        ]
        concat_nonce = nonce + recieved_nonce

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(concat_nonce)
        tag = mac.finalize()

        # Construct nonce message
        nonce_msg = nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def send_nonce_msg_to_host(self, connection, prederived_key_hash, derived_key):
        # Generate Nonce
        nonce = os.urandom(self.nonce_byte_size)

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(nonce)
        tag = mac.finalize()

        # Create key confirmation message
        nonce_msg = prederived_key_hash + nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def verify_mac_from_host(self, recieved_nonce_msg, generated_nonce, derived_key):
        success = False

        recieved_nonce = recieved_nonce_msg[0 : self.nonce_byte_size]

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce + generated_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[self.nonce_byte_size :]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success

    # TODO: Already refactored by putting it in common_protocols.py, delete and change all reference from this to common_protocols.py version
    def verify_mac_from_device(
        self, recieved_nonce_msg, derived_key, prederived_key_hash
    ):
        success = False

        # Retrieve nonce used by device
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len : pd_hash_len + self.nonce_byte_size
        ]

        # Generate new MAC tag for the nonce with respect to the derived key
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[pd_hash_len + self.nonce_byte_size :]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success


"""###TESTING CODE###
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
    from test_sensor import Test_Sensor
    from sensor_reader import Sensor_Reader
    prot = Miettinen_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024)),
        8,
        4,
        1,
        1,
        0.003,
        2.97e-05,
        0.9,
        5,
        20,
        10)
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()"""
