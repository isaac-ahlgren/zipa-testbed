# TODO compare with Seemoo Lab implementation: https://github.com/seemoo-lab/ubicomp19_zero_interaction_security/blob/master/Visualization/SchuermannSigg.ipynb
import math
import multiprocessing as mp
import time

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes

from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj
from networking.network import *


# TODO: Make template for protocols so there is are guaranteed boiler plate functionality in how to initialize it
class Shurmann_Siggs_Protocol:
    def __init__(
        self,
        sensor,
        key_length,
        parity_symbols,
        window_len,
        band_len,
        timeout,
        logger,
        verbose=True,
    ):
        self.sensor = sensor
        self.window_len = window_len
        self.band_len = band_len

        self.name = "shurmann-siggs"
        self.timeout = timeout

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, key_length), key_length
        )
        self.hash_func = hashes.SHA256()

        # Conversion from how many requested bits you need to how much sample data you will need for that
        self.time_length = (
            math.ceil(
                ((self.commitment_length * 8) / int((window_len / 2 + 1) / band_len))
                + 1
            )
            * window_len
        )

        self.logger = logger

        self.count = 0

        self.verbose = verbose

    def sigs_algo(self, x1, window_len=10000, bands=1000):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        FFTs = []
        from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x)

        x = np.array(x1.copy())
        # wind = scipy.signal.windows.hann(window_len)
        for i in range(0, len(x), window_len):
            if len(x[i : i + window_len]) < window_len:
                # wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i : i + window_len] = x[i : i + window_len]  # * wind
            else:
                x[i : i + window_len] = x[i : i + window_len]  # * wind

            FFTs.append(abs(rfft(x[i : i + window_len])))

        E = {}
        bands_lst = []
        for i in range(0, len(FFTs)):
            frame = FFTs[i]
            bands_lst.append(
                [frame[k : k + bands] for k in range(0, len(frame), bands)]
            )
            for j in range(0, len(bands_lst[i])):
                E[(i, j)] = np.sum(bands_lst[i][j])

        bs = ""

        count = 0
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if E[(i, j)] - E[(i, j + 1)] - (E[(i - 1, j)] - E[(i - 1, j + 1)]) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)

    def extract_context(self):
        signal = self.sensor.read(self.time_length)
        bits = self.sigs_algo(signal, window_len=self.window_len, bands=self.band_len)
        return bits, signal

    def parameters(self, is_host):
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"window_length: {self.window_len}\n"
        parameters += f"band_length: {self.band_len}\n"
        parameters += f"time_length: {self.time_length}\n"

    def device_protocol(self, host):
        host.setblocking(1)
        if self.verbose:
            print(f"Iteration {str(self.count)}.\n")

        # Log parameters to NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("Sending ACK.\n")
        ack(host)

        # Wait for ack from host to begin context extract, quit early if no response within time
        if self.verbose:
            print("Waiting for ACK from host.")
        if not ack_standby(host, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return

        # Extract bits from mic
        if self.verbose:
            print("Extracting context\n")
        witness, signal = self.extract_context()

        if self.verbose:
            print("witness: " + str(witness))

        # Wait for Commitment
        if self.verbose:
            print("Waiting for commitment from host")
        commitments, recieved_hashes = commit_standby(host, self.timeout)

        commitment = commitments[0]
        recieved_hash = recieved_hashes[0]

        # Early exist if no commitment recieved in time
        if not commitment:
            if self.verbose:
                print("No commitment recieved within time limit - early exit\n")
            return

        # Decommit
        if self.verbose:
            print("Decommiting")
        key = self.re.decommit_witness(commitment, witness)

        generated_hash = self.hash_function(key)

        success = False
        if constant_time.bytes_eq(generated_hash, recieved_hash):
            success = True

        if self.verbose:
            print(f"key: {str(key)}\n success: {str(success)}\n")

        self.logger.log(
            [
                ("witness", "txt", witness),
                ("commitment", "txt", commitment),
                ("success", "txt", str(success)),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
        )

        self.count += 1

    def host_protocol(self, device_sockets):
        # Log parameters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count))
            print()
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket):
        # Exit early if no devices to pair with
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return
        if self.verbose:
            print("Successfully ACKed participating device")
            print()

        if self.verbose:
            print("ACKing all participating devices")
        ack(device_socket)

        # Extract key from mic
        if self.verbose:
            print("Extracting Context\n")
        witness, signal = self.extract_context()

        # Commit Secret
        if self.verbose:
            print("Commiting Witness")

        secret_key, commitment = self.re.commit_witness(witness)

        if self.verbose:
            print("witness: " + str(witness))

        hash = self.hash_function(secret_key)

        if self.verbose:
            print("Sending commitment")
            print()
        send_commit([commitment], [hash], device_socket)

        self.logger.log(
            [
                ("witness", "txt", str(witness)),
                ("commitment", "txt", commitment),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
        )

        self.count += 1

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()


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
    prot = Shurmann_Siggs_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024)),
                                   8,
                                   4,
                                   10000,
                                   1000,
                                   10,
                                   None,
    )
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()"""
