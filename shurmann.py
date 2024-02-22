import time
import multiprocessing as mp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import constant_time
import numpy as np
from reed_solomon import ReedSolomonObj
from corrector import Fuzzy_Commitment
from network import *

# TODO: Make it so that key length is a parameter to the Shurmann and siggs protocol
# It will need work with how many bits the quantizer
# TODO: Make template for protocols so there is are guaranteed boiler plate functionality in how to initialize it
# TODO: Make it so window_len and bands are parameterized
class Shurmann_Siggs_Protocol:
    def __init__(self, sensor, n, k, timeout, nfs_server_dir, identifier):
        self.sensor = sensor
        self.re = Fuzzy_Commitment(ReedSolomonObj(n ,k), 8)
        self.name = "shurmann-siggs"
        self.count = 0
        self.timeout = timeout

        self.n = n
        self.k = k

        self.hash_func = hashes.SHA512() # Can't change this without breaking the commit network function (will fix later)

        # These variables should be in the NFS object
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier

    def sigs_algo(self, x1, window_len=10000, bands=1000):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
        
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
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if E[(i, j)] - E[(i, j + 1)] - (E[(i - 1, j)] - E[(i - 1, j + 1)]) > 0:
                    bs += "1"
                else:
                    bs += "0"

        return bitstring_to_bytes(bs)

    def extract_context(self):
        signal = self.sensor.read()
        bits = self.sigs_algo(signal)
        return bits, signal

    def device_protocol(self, host):
        host.setblocking(1)
        print(f"Iteration {str(self.count)}.\n")

        # Sending ack that they are ready to begin
        print("Sending ACK.\n")
        ack(host)

        # Wait for ack from host to begin context extract, quit early if no response within time
        print("Waiting for ACK from host.")
        if not ack_standby(host, self.timeout):
            print("No ACK recieved within time limit - early exit.\n\n")
            return

        # Extract bits from mic
        print("Extracting context\n")
        witness, signal = self.extract_context()

        # Wait for Commitment
        print("Waiting for commitment from host")
        commitment, recieved_hash = commit_standby(host, self.timeout)

        # Early exist if no commitment recieved in time
        if not commitment:
            print("No commitment recieved within time limit - early exit\n")
            return

        # Decommit
        print("Decommiting")
        key = self.re.decommit_witness(commitment, witness)

        generated_hash = self.hash_function(key)
 
        success = False
        if constant_time.bytes_eq(generated_hash, recieved_hash):
            success = True

        print(f"key: {str(key)}\n success: {str(success)}\n")

        self.count += 1

    def host_protocol(self, device_sockets):
        print("Iteration " + str(self.count))
        print()
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket):

        # Exit early if no devices to pair with
        if not ack_standby(device_socket, self.timeout):
            print("No ACK recieved within time limit - early exit.\n\n")
            return
        print("Successfully ACKed participating device")
        print()

        print("ACKing all participating devices")
        ack(device_socket)

        # Extract key from mic
        print("Extracting Context")
        witness, signal = self.extract_context()
        print()

        # Commit Secret
        print("Commiting Witness")
        secret_key, commitment = self.re.commit_witness(witness)

        hash = self.hash_function(secret_key)

        print("Sending commitment")
        print()
        send_commit(commitment, hash, device_socket)

        self.count += 1

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()


'''###TESTING CODE###
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
    prot = Shurmann_Siggs_Protocol(None, 12, 8, 10, None, None)
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()'''