import numpy as np
from microphone import Microphone
from corrector import Fuzzy_Commitment
from galois import *
from network import *

class Shurmann_Siggs_Protocol():
    def __init__(self, microphone, n, k, timeout):
        self.signal_measurement = microphone 
        self.re = Fuzzy_Commitment(n, k)
        self.protocol_name = "shur_n_sigg"
        self.count = 0
        self.timeout = timeout

    def sigs_algo(self, x1,window_len=10000, bands=1000):
        FFTs = []
        from scipy.fft import fft, fftfreq, ifft, rfft, irfft

        if window_len == 0:
            window_len = len(x)

        x = np.array(x1.copy())
        #wind = scipy.signal.windows.hann(window_len)
        for i in range(0,len(x),window_len):
        
            if len(x[i:i+window_len]) < window_len:
                #wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i:i+window_len] = x[i:i+window_len] #* wind
            else:
                x[i:i+window_len] = x[i:i+window_len] #* wind

            FFTs.append(abs(rfft(x[i:i+window_len])))
 
        E = {}
        bands_lst = []
        for i in range(0,len(FFTs)):
            frame = FFTs[i]
            bands_lst.append([ frame[k:k+bands] for k in range(0,len(frame),bands)])
            for j in range(0,len(bands_lst[i])):
                E[(i,j)] = np.sum(bands_lst[i][j])

        bs = ""
        for i in range(1,len(FFTs)):
            for j in range(0,len(bands_lst[i])-1):
            
                if E[(i,j)] -E[(i,j+1)] - (E[(i-1,j)] - E[(i-1,j+1)]) > 0:
                    bs+= "1"
                else:
                    bs+="0"   
        return bs

    def extract_context(self):
        print()
        print("Extracting Context")
        signal = self.signal_measurement.get_audio()
        bits = self.sigs_algo(signal)
        print()
        return bits, signal

    def device_protocol(self, host_socket):
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
        print()

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
