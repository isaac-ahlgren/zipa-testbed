import numpy as np
from corrector import Fuzzy_Commitment
from galois import *
from network import *

class Miettinen_Protocol():
    def __init__(self, n, k, f, w, rel_thresh, abs_thresh):
        self.n = n
        self.k = k
        self.f = f
        self.w = w
        self.rel_thresh = rel_thresh
        self.abs_thresh = abs_thresh
        self.re = Fuzzy_Commitment(n, k)

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
        signal = signal_preprocessing(x, self.f, self.w)
        key = gen_key(x, self.rel_thresh, self.abs_thresh)
        return key

    def decommit_witness(self, commitment, witness, h):
        C, success = self.re.decommit_witness(commitment, witness, h)
        return C, success

    def commit_witness(self, witness):
        secret_key, h, commitment = self.re.commit_witness(witness)
        return secret_key, h, commitment

    def extract_context(self):
        print()
        print("Extracting Context")
        signal = self.signal_measurement.get_audio()
        bits = self.miettinen_algo(signal)
        print()
        return bits, signal

    # implementing this in a bit
    def key_strengthening_process(self):
        return

    def device_protocol(self, host_socket):
        host_socket.setblocking(1)
        print("Iteration " + str(self.count))
            
        # Sending ack that they are ready to begin
        print()
        print("Sending ACK")
        ack(host_socket)

        # Wait for ack from host to being context extract, quit early if no response within time
        print()
        print("Waiting for ACK from host")
        if not wait_for_ack(host_socket, self.timeout):
            print("No ACK recieved within time limit - early exit")
            print()
            return
        print()

        # Extract bits from mic
        print("Extracting context")
        print()
        witness, signal = self.extract_context()
        
        # Wait for Commitment
        print("Waiting for commitment from host")
        commitment, h = wait_for_commitment(host_socket, self.timeout)

        # Early exist if no commitment recieved in time
        if not commitment:
            print("No commitment recieved within time limit - early exit")
            print()
            return
        print()

        print("witness: " + str(hex(int(witness, 2))))
        print("h: " + str(h))
        print()

        # Decommit
        print("Decommiting")
        C, success = self.re.decommit_witness(commitment, witness, h)

        print("C: " + str(C))
        print("success: " + str(success))
        print()

        # Log all information to NFS server
        print("Logging all information to NFS server")
        self.send_to_nfs_server("audio", signal, witness, h, commitment)

        self.count += 1

    def host_protocol(self, device_sockets):
        print("Iteration " + str(self.count))
        print()
  
        participating_sockets = wait_for_all_ack(device_sockets, self.timeout)

        # Exit early if no devices to pair with
        if len(participating_sockets) == 0:
            print("No advertised devices joined the protocol - early exit")
            print()
            return
        print("Successfully ACKed participating devices")
        print()
        

        print("ACKing all participating devices")
        ack_all(participating_sockets)

        # Extract key from mic
        print("Extracting Context")
        witness, signal = self.extract_context()
        print()

        # Commit Secret
        print("Commiting Witness")
        secret_key, h, commitment = self.re.commit_witness(witness)

        print("witness: " + str(hex(int(witness, 2))))
        print("h: " + str(h))
        print()

        print("Sending commitment")
        print()
        send_commitment(commitment, h, participating_sockets)

        # Log all information to NFS server
        print("Logging all information to NFS server")
        self.send_to_nfs_server("audio", signal, witness, h, commitment)

        self.count += 1

    def send_to_nfs_server(self, signal_type, signal, witness, h, commitment):
        root_file_name = self.nfs_server_dir + "/" + signal_type

        signal_file_name = root_file_name + "_signal_id" + str(self.identifier) + "_it" + str(self.count) + ".csv"
        witness_file_name = root_file_name + "_witness_id" + str(self.identifier) + "_it" + str(self.count) + ".txt"
        hash_file_name = root_file_name + "_hash_id" + str(self.identifier) + "_it" + str(self.count) + ".txt"
        commitment_file_name = root_file_name + "_commitment_id" + str(self.identifier) + "_it" + str(self.count) + ".csv"

        np.savetxt(signal_file_name, signal)
        np.savetxt(commitment_file_name, np.array(commitment.coeffs))
        with open(witness_file_name, "w") as text_file:
            text_file.write(witness)
        with open(hash_file_name, "w") as text_file:
            text_file.write(str(h))

