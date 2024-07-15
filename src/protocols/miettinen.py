import queue

import numpy as np
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from networking.network import (
    VERBOSE_MESSAGES,
    ack,
    ack_standby,
    commit_standby,
    get_nonce_msg_standby,
    send_commit,
)
from protocols.common_protocols import (
    diffie_hellman,
    send_nonce_msg_to_device,
    send_nonce_msg_to_host,
    verify_mac_from_device,
    verify_mac_from_host,
)
from protocols.protocol_interface import ProtocolInterface


class Miettinen_Protocol(ProtocolInterface):
    def __init__(self, parameters, sensor, logger):
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "Miettinen_Protocol"
        self.wip = False
        self.f = int(parameters["f"] * self.sensor.sensor.sample_rate)
        self.w = int(parameters["w"] * self.sensor.sensor.sample_rate)
        self.rel_thresh = parameters["rel_thresh"]
        self.abs_thresh = parameters["abs_thresh"]
        self.auth_thresh = parameters["auth_thresh"]
        self.success_thresh = parameters["success_thresh"]
        self.max_iterations = parameters["max_iterations"]
        self.nonce_byte_size = 16
        self.time_length = (self.w + self.f) * (self.commitment_length * 8 + 1)
        self.count = 0

    def signal_preprocessing(signal, no_snap_shot_width, snap_shot_width):
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

    def gen_key(c, rel_thresh, abs_thresh):
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
    def miettinen_algo(self, x, f, w, rel_thresh, abs_thresh):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        signal = self.signal_preprocessing(x, f, w)
        key = self.gen_key(signal, rel_thresh, abs_thresh)

        return bitstring_to_bytes(key)

    def extract_context(self):
        signal = []

        while True:
            try:
                data = self.queue.get()
                signal.extend(data)
            except queue.Empty:
                continue

            if len(signal) >= int(self.time_length * self.sensor.sensor.sample_rate):
                self.flag.value = -1
                break

        bits = self.miettinen_algo(
            signal, self.f, self.w, self.rel_thresh, self.abs_thresh
        )
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
        parameters += "\nsuccess_threshold: " + str(self.success_thresh)
        parameters += "\nauthentication_threshold: " + str(self.auth_thresh)
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
            print(VERBOSE_MESSAGES["ACK_SEND"])
        ack(host_socket)

        # Wait for ack from host to being context extract,
        # quit early if no response within time
        if self.verbose:
            print(VERBOSE_MESSAGES["ACK_WAIT"])

        if not ack_standby(host_socket, self.timeout):
            if self.verbose:
                print(VERBOSE_MESSAGES["ACK_TIMEOUT"])
            return

        # Shared key generated
        shared_key = diffie_hellman(host_socket)

        current_key = shared_key
        successes = 0
        total_iterations = 0
        while (
            successes < self.success_thresh and total_iterations < self.max_iterations
        ):
            # Sending ack that they are ready to begin

            if self.verbose:
                print(VERBOSE_MESSAGES["ACK_WAIT"])
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print(VERBOSE_MESSAGES["ACK_TIMEOUT"])
                return

            if self.verbose:
                print(VERBOSE_MESSAGES["ACK_SEND"])
            ack(host_socket)

            success = False

            # Extract bits from sensor
            witness, signal = self.extract_context()

            # Wait for Commitment
            if self.verbose:
                print(VERBOSE_MESSAGES["COMM_WAIT"])
            commitments, hs = commit_standby(host_socket, self.timeout)

            commitment = commitments[0]

            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print(VERBOSE_MESSAGES["COMM_TIMEOUT"])
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
            generated_nonce = send_nonce_msg_to_host(
                host_socket, pd_key_hash, derived_key
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print(VERBOSE_MESSAGES["NONC_TIMEOUT"])
                return

            # If hashes are equal, then it was successful
            if verify_mac_from_host(recieved_nonce_msg, generated_nonce, derived_key):
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
            if successes / total_iterations >= self.auth_thresh:
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
                    + str(successes / total_iterations >= self.auth_thresh),
                )
            ]
        )

        self.count += 1

    def host_protocol_single_threaded(self, device_socket):
        device_socket.setblocking(1)

        device_ip_addr, device_port = device_socket.getpeername()

        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print(VERBOSE_MESSAGES["ACK_TIMEOUT"])
            return

        if self.verbose:
            print(VERBOSE_MESSAGES["ACK_ALL"])
        ack(device_socket)

        # Shared key generated
        shared_key = diffie_hellman(device_socket)

        current_key = shared_key
        total_iterations = 0
        successes = 0
        while (
            successes < self.success_thresh and total_iterations < self.max_iterations
        ):
            success = False
            # ACK device
            if self.verbose:
                ack(device_socket)

            if not ack_standby(device_socket, self.timeout):
                if self.verbose:
                    print(VERBOSE_MESSAGES["ACK_TIMEOUT"])
                return

            if self.verbose:
                print(VERBOSE_MESSAGES["ACK_SUCCESS"])

            # Extract key from sensor
            witness, signal = self.extract_context()

            # Commit Secret
            if self.verbose:
                print(VERBOSE_MESSAGES["COMM_WITNESS"])
            prederived_key, commitment = self.re.commit_witness(witness)

            if self.verbose:
                print("witness: " + str(witness))
                print()

            if self.verbose:
                print(VERBOSE_MESSAGES["COMM_SEND"])
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
                    print(VERBOSE_MESSAGES["NONC_TIMEOUT"])
                return

            # Derive new key using previous key and new prederived key
            # from fuzzy commitment
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            if verify_mac_from_device(recieved_nonce_msg, derived_key, pd_key_hash):
                success = True
                successes += 1
                current_key = derived_key

            # Create and send key confirmation value
            send_nonce_msg_to_device(
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
            if successes / total_iterations >= self.auth_thresh:
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
                    + str(successes / total_iterations >= self.auth_thresh),
                )
            ],
            ip_addr=device_ip_addr,
        )

        self.count += 1


"""
###TESTING CODE###
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
    prot = Miettinen_Protocol(
        Sensor_Reader(
            Test_Sensor(
                44100,
                44100*400,
                1024)),
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
    d.start()
"""
