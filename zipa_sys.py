from network import Network
from corrector import Fuzzy_Commitment
from galois import *
from shurmann import sigs_algo
from microphone import Microphone
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import ipaddress
import socket
import time
import sys
import pickle
import numpy as np

# Device communcation identifiers
HOST = "host    "  
START = "start   "
ACK = "ack     "
COMMITMENT = "comm    "

class ZIPA_System():
    def __init__(self, identifier, ip, port, service_name, nfs_server_dir, timeout, sample_rate, seconds, n, k):
        self.identifier = identifier
        self.nfs_server_dir = nfs_server_dir
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.timeout = timeout

        self.net = Network(ip, port, service_name)
        self.signal_measurement = Microphone(sample_rate, int(seconds*sample_rate)) 
        self.re = Fuzzy_Commitment(n, k)

        self.count = 0

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
 
    def extract_context(self):
        print()
        print("Extracting Context")
        signal = self.signal_measurement.get_audio()
        bits = sigs_algo(signal)
        print()
        return bits, signal

    def zipa_protocol(self, uncond_host):
        while 1:
            if uncond_host:
                print("Unconditionally being host")
                self.host_protocol()
            else:
                msg = self.net.get_msg()
                if msg:
                    ip_addr = msg[0]
                    data = msg[1]
                    if data.decode() == HOST:
                        print("Starting protocol as host")
                        print()
                        self.host_protocol()
                    elif data.decode() == START:
                        print("Starting protocol as device")
                        print()
                        self.device_protocol(ip_addr)

    def device_protocol(self, host_ip):
        print("Iteration " + str(self.count))
            
        # Sending ack that they are ready to begin
        print()
        print("Sending ACK")
        self.net.send_ack()

        # Wait for ack from host to being context extract, quit early if no response within time
        print()
        print("Waiting for ACK from host")
        if not self.wait_for_ack(host_ip):
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
        commitment, h = self.wait_for_commitment()

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

    def host_protocol(self):
        print("Iteration " + str(self.count))
        print()

        # Send and confirm all devices in the protocol
        print("Initializing protocol for all advertising devices")
        confirmed_ip_addrs = self.initialize_protocol()

        # Exit early if no devices to pair with
        if len(confirmed_ip_addrs) == 0:
            print("No advertised devices joined the protocol - early exit")
            print()
            return
        print()

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

        # Sending commitment
        print("Sending commitment to all devices")
        self.send_commitment(commitment, h, confirmed_ip_addrs)
        print()

        print("Waiting for devices to ACK")
        acked_ip_addrs = self.wait_for_all_ack(confirmed_ip_addrs)

        # Exit early if no devices in protocol
        if len(acked_ip_addrs) == 0:
            print("No devices ACKed - early exit")
            return

        # Log all information to NFS server
        print("Logging all information to NFS server")
        self.send_to_nfs_server("audio", signal, witness, h, commitment)

        self.count += 1

    def ack(self, host_ip_addr):
        self.net.send_msg(ACK.encode(), host_ip_addr)

    def wait_for_ack(self, host_ip_addr):
        acked = False

        start_time = time.time()
        current_time = start_time
        while (current_time - start_time) >= self.timeout:
            current_time = time.time()
            msg = self.net.get_msg()
            if msg == None:
                continue
            else:
                ip_addr = msg[0]
                data = msg[1]
                if data.decode() == ACK and host_ip_addr == ip_addr:
                    acked = True
                    break

        return acked

    def ack_all(self, confirmed_ip_addrs):
        for i in range(len(confirmed_ip_addrs)):
            self.net.send_msg(ACK.encode(), confirmed_ip_addrs)

    def wait_for_all_ack(self, potential_ip_addrs):
        acked_ip_addrs = []

        start_time = time.time()
        current_time = start_time
        while (current_time - start_time) >= self.timeout:
            if len(potential_ip_addrs) == len(acked_ip_addrs):
                break

            current_time = time.time()
            msg = self.net.get_msg()
            if msg == None:
                continue
            else:
                ip_addr = msg[0]
                data = msg[1]
                if data.decode() == ACK and ip_addr in potential_ip_addr:
                    acked_ip_addrs.append(ip_addr)

        return acked_ip_addrs

    def initialize_protocol(self):
        potential_ip_addrs = self.net.get_zipa_ip_addrs()
        print("Potential ip addrs: " + str(potential_ip_addrs))

        # Send a start message to all available protocols
        for i in range(len(potential_ip_addrs)):
            self.net.conn_to(potential_ip_addrs[i])
            self.net.send_msg(START.encode(), potential_ip_addrs[i])

        # Poll for devices for an ack response for at least timeout seconds
        acked_ip_addrs = self.wait_for_all_ack(potential_ip_addrs)

        # Ack all devices in the protocol to begin context extraction
        self.ack_all(acked_ip_addrs)

        return acked_ip_addrs

    def send_commitment(self, commitment, h, confirmed_ip_addrs):
        pickled_comm = pickle.dumps(commitment)
        msg = COMM.encode() + h + pickled_comm
        for i in range(len(confirmed_ip_addrs)):
            self.net.send_msg(msg, confirmed_ip_addrs[i])

    def wait_for_commitment(self, host_ip_addr):
        commitment = None
        h = None
        
        start_time = time.time()
        current_time = start_time
        while (current_time - start_time) >= self.timeout:
            current_time = time.time()
            msg = self.net.get_msg()
            if msg == None:
                continue
            else:
                ip_addr = msg[0]
                data = msg[1]
                if msg[:8].decode() == COMM and ip_addr == host_ip_addr:
                    h = msg[8:72] # 64 byte hash
                    commitment = msg[72:]

        return commitment, h
