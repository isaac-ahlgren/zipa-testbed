import ipaddress
import pickle
import socket
from multiprocessing import Process

import numpy as np

from browser import ZIPA_Service_Browser
from microphone import Microphone
from network import *
from shurmann import Shurmann_Siggs_Protocol

# Used for sending a message to the devices, ID'ing the host and the process is concatenated to this
HOST = "host    "
START = "start   "


class ZIPA_System:
    def __init__(
        self,
        identifier,
        ip,
        port,
        service_name,
        nfs_server_dir,
        timeout,
        sample_rate,
        seconds,
        n,
        k,
    ):
        # Object is created and filled with identifying info as well as setup info
        self.identifier = identifier
        self.nfs_server_dir = nfs_server_dir
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.timeout = timeout

        # Setup listening socket
        self.ip = ip
        self.port = port
        # Create a TCP socket that can communicate through IPv4.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set socket level option to broadcast a message to devices on the network
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Socket can be reused even if in the TIME_WAIT state, that waits for stragglers before closing connection
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Socket will listen on the socket of specified IP address
        self.sock.bind((ip, port))
        # Continues to run, doing other work, even if the socket operation isn't completed
        self.sock.setblocking(0)
        # Listens for incoming connections
        self.sock.listen()

        # Socket list and add itself to the list of open sockets
        self.open_socks = [self.sock]

        # Setup service browser thread
        self.browser = ZIPA_Service_Browser(ip, service_name)

        # Setup sensors
        self.microphone = Microphone(sample_rate, int(seconds * sample_rate))

        # Setup protocols
        self.protocols = [
            Shurmann_Siggs_Protocol(
                self.microphone, n, k, timeout, nfs_server_dir, identifier
            )
        ]
        self.protocol_threads = []

    def start(self):
        # Begin looking for devices that are advertising ZIPA
        print("Starting browser thread")
        print()
        self.browser.start_thread()

        #
        print("Starting listening thread")
        print()
        outputs = []
        # Running until the end of time
        while 1:
            # Keep track of incoming data, data ready to send, and data that causes errors
            readable, writable, exceptional = select.select(
                self.open_socks, outputs, self.open_socks
            )
            # When the host device is receiving data from other devices
            for s in readable:
                # If this is the first time connecting to this client
                if s is self.sock:
                    # Accept the connection
                    connection, client_address = self.sock.accept()
                    # Allow for multiple connections from other clients
                    connection.setblocking(0)
                    print("Connection Made: " + str(client_address))
                    print()
                    # Add to the list of available ZIPA devices
                    self.open_socks.append(connection)
                # Otherwise if we already have a connection to a device
                else:
                    # Begin receiving the data from the client
                    data = s.recv(1024)

                    # If non-empty data is recieved
                    if data:
                        # Determine the type of message that was received
                        self.service_request(data, s)
                    else:
                        # Otherwise the process is complete and we close the connection
                        self.open_socks.remove(s)
                        s.close()

            # For sockets whose connection couldn't be established
            for s in exceptional:
                # Remove and close socket
                self.open_socks.remove(s)
                s.close()

    # Receives what device and what protocol to perform
    def service_request(self, data, sock):
        # From server.py file
        msg = data.decode()
        if msg[:8] == HOST:
            # In the list of protocols
            for i in range(len(self.protocols)):
                # Find the protocol that the message wants to do
                if self.protocols[i].name == msg[8:]:
                    protocol_name = msg[8:]
                    participating_sockets = self.initialize_protocol(protocol_name)

                    # Abort if there's no participants
                    if len(participating_sockets) == 0:
                        print("No advertised devices joined the protocol - early exit")
                        return False
                    print()

                    # Stage the specified protocol to all participating devices on its own thread
                    new_thread = Process(
                        target=self.protocols[i].host_protocol(participating_sockets)
                    )
                    new_thread.start()
                    self.protocol_threads.append(new_thread)
        elif msg[:8] == START:
            for i in range(len(self.protocols)):
                if self.protocols[i].name == msg[8:]:
                    protocol_name = msg[8:]
                    # Target a device to perform the protocol
                    new_thread = Process(target=self.protocols[i].device_protocol(sock))
                    new_thread.start()
                    # Remove from available devices as it's currently running a protocol
                    self.open_socks.remove(sock)
                    # Note the thread that it's working on the protocol
                    self.protocol_threads.append(new_thread)

    def initialize_protocol(self, protocol_name):
        print("Initializing protocol for all advertising devices")
        potential_ip_addrs = self.browser.get_ip_addrs_for_zipa()
        print("Potential ip addrs: " + str(potential_ip_addrs))

        participating_ip_addrs = []
        # Send a start message to all available protocols
        for i in range(len(potential_ip_addrs)):
            # Create a new TCP socket that services IPv4 addresses
            new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set the socket to broadcast on the network and can be reused at any time
            new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # For devices that couldn't connect to perform the protocol
            err = new_sock.connect_ex((potential_ip_addrs[i], self.port))
            # If we connected to a client successfully
            if not err:
                # Call for the start of the protocol and send to the client
                msg = (START + protocol_name).encode()
                new_sock.send(msg)
                # Note the socket that's performing the protocol
                participating_ip_addrs.append(new_sock)
            else:
                new_sock.close()
                print(
                    "Error connecting to " + potential_ip_addrs[i] + " err: " + str(err)
                )
                print()
        # List all participating devices working on the protocol
        return participating_ip_addrs
