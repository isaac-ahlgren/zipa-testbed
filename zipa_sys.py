from network import *
from browser import ZIPA_Service_Browser
from shurmann import Shurmann_Siggs_Protocol
from microphone import Microphone
from multiprocessing import Process
import ipaddress
import socket
import pickle
import numpy as np

HOST = "host    "  
START = "start   "
class ZIPA_System():
    def __init__(self, identifier, ip, port, service_name, nfs_server_dir, timeout, sample_rate, seconds, n, k):
        self.identifier = identifier
        self.nfs_server_dir = nfs_server_dir
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.timeout = timeout

        # Setup listening socket
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((ip, port))
        self.sock.setblocking(0)
        self.sock.listen()

        # Socket list
        self.open_socks = [self.sock]

        # Setup service browser thread
        self.browser = ZIPA_Service_Browser(ip, service_name)

        # Setup sensors
        self.microphone = Microphone(sample_rate, int(seconds*sample_rate))

        # Setup protocols
        self.protocols = [Shurmann_Siggs_Protocol(self.microphone, n, k, timeout)] 
        self.protocol_threads = []

    def start(self):
        print("Starting browser thread")
        print()
        self.browser.start_thread()

        print("Starting listening thread")
        print()
        outputs = []
        while (1):
            readable, writable, exceptional = select.select(self.open_socks, outputs, self.open_socks)
            for s in readable:
                if s is self.sock:
                    connection, client_address = self.sock.accept()
                    connection.setblocking(0)
                    self.open_socks.append(connection)
                else:
                    data = s.recv(1024)
                    if data:
                        self.service_request(data, s)
                    else:
                        self.open_socks.remove(s)
                        s.close()

            for s in exceptional:
                self.open_socks.remove(s)
                s.close()

    def service_request(self, data, sock):
        msg = data.decode()
        if msg[:8] == HOST:
            for i in range(len(self.protocols)):
                if self.protocols[i].protocol_name == msg[8:]:
                    protocol_name = msg[8:]
                    participating_sockets = self.initialize_protocol(protocol_name)
                    
                    if len(participating_sockets) == 0:
                        print("No advertised devices joined the protocol - early exit")
                        return False
                    print()

                    new_thread = Process(target=self.protocols[i].host_protocol(participating_sockets))
                    new_thread.start()
                    self.protocol_threads.append(new_thread)
        elif msg[:8] == START:
            for i in range(len(self.protocols)):
                if self.protocols[i].protocol_name == msg[8:]:
                    protocol_name = msg[8:]
                    new_thread = Process(target=self.protocols[i].device_protocol(sock))
                    new_thread.start()
                    self.open_socks.remove(s)
                    self.protocol_threads.append(new_thread)
 
    def initialize_protocol(self, protocol_name):
        print("Initializing protocol for all advertising devices")
        potential_ip_addrs = self.browser.get_ip_addrs_for_zipa()
        print("Potential ip addrs: " + str(potential_ip_addrs))

        participating_ip_addrs = []
        # Send a start message to all available protocols
        for i in range(len(potential_ip_addrs)):
            new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            err = new_sock.connect_ex((potential_ip_addrs[i], self.port))
            if not err:
                msg = (START + protocol_name).encode()
                new_sock.send(msg)
                participating_ip_addrs.append(new_sock)
            else:
                new_sock.close()
                print("Error connecting to " + potential_ip_addrs[i] + " err: " + str(err))
                print()

        return participating_ip_addrs
