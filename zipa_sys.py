from network import Ad_Hoc_Network
from corrector import Fuzzy_Commitment
from galois import *
from shurmann import sigs_algo
from microphone import Microphone
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from multiprocessing import Process, Lock, shared_memory
import ipaddress
import socket
import time
import sys
import numpy as np

class ZIPA_System():
    def __init__(self, identifier, is_host, ip, other_ip, nfs_server_dir, sample_rate, seconds, exp_name, n, k):
        # Setup for main thread
        self.identifier = identifier
        self.nfs_server_dir = nfs_server_dir
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.net = Ad_Hoc_Network(ip, other_ip, is_host)
        self.signal_measurement = Microphone(sample_rate, int(seconds*sample_rate)) 
        self.re = Fuzzy_Commitment(n, k)
        self.exp_name = exp_name
        self.count = 0

        # Setup for service browser thread
        zeroconf = Zeroconf()
        listener = ZIPA_Service_Listener()
        browser = ServiceBrowser(zeroconf, "_zipa._tcp.local.", listener)
        serv_browser_thread = Process(target=browser.run)
        serv_browser_thread.start()

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

    def device_protocol(self): 
        while (1):
            print("Iteration " + str(self.count))

            # Wait for start from host
            self.net.get_start()
            
            # Sending ack that they can start
            self.net.send_ack()

            # Extract bits from mic
            witness, signal = self.extract_context()

            # Wait for Commitment
            commitment, h = self.net.get_commitment(8192)

            print("witness: " + str(hex(int(witness, 2))))
            print("h: " + str(h))
            # Decommit
            C, success = self.re.decommit_witness(commitment, witness, h)
            print("C: " + str(C))
            print("success: " + str(success))
            print()

            # Log all information to NFS server
            self.send_to_nfs_server("audio", signal, witness, h, commitment)

            self.count += 1

    def host_protocol(self):
        while (1):
            print("Iteration " + str(self.count))

            # Send start to device
            self.net.send_start()
            
            # Get Ack to make sure it isn't lagging from the previous iteration
            self.net.get_ack()

            # Extract key from mic
            witness, signal = self.extract_context()

            # Commit Secret
            secret_key, h, commitment = self.re.commit_witness(witness)

            print("witness: " + str(hex(int(witness, 2))))
            print("h: " + str(h))
            print()
            # Sending Codeword
            self.net.send_commitment(commitment, h)

            # Log all information to NFS server
            self.send_to_nfs_server("audio", signal, witness, h, commitment)

            self.count += 1

class ZIPA_Service_Listener(ServiceListener):
    def __init__(self):
        self.addr_list_len = 256
        self.zipa_addrs = shared_memory.ShareableList([0 for i in range(self.addr_list_len)])
        self.mutex = Lock()

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.mutex.acquire()
        ip_addr = int(ipaddress.ip_address(socket.gethostbyname(name[:name.index('.')] + ".local")))
        for i in range(self.addr_list_len):
            if ip_addr == self.zipa_addrs[i]:
                self.zipa_addrs[i] = 0
        self.mutex.release()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        return

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.mutex.acquire()
        print(name)
        ip_addr = int(ipaddress.ip_address(socket.gethostbyname(name[:name.index('.')] + ".local")))
        for i in range(self.addr_list_len):
            if self.zipa_addrs[i] == 0:
                self.zipa_addrs[i] = ip_addr
        self.mutex.release()



