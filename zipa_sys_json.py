import json
import socket
import pickle
import select
from multiprocessing import Process

from browser import ZIPA_Service_Browser
from microphone import Microphone
# from network import *
# from network_json import *
from shurmann import Shurmann_Siggs_Protocol

# Used to initiate and begin protocol
HOST = "host    "
STRT = "start   "


class ZIPA_System:
    def __init__(self, identity, ip, port, service, nfs, timeout, sampling, time, n, k):
        # Object is created and filled with identifying info as well as setup information
        self.id = identity
        self.nfs = nfs
        self.sampling = sampling
        self.time = time
        self.timeout = timeout

        # Set up a listening socket
        self.ip = ip
        self.port = port
        # Create a reusable TCP socket through IPv4 broadcasting on the network
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind socket to the specified socket and IPv4 address that runs continuously
        self.socket.bind(ip, port)
        self.socket.setblocking(0)
        self.socket.listen()

        # Set up a discovery for the protocol
        self.discoverable = [self.socket]
        self.browser = ZIPA_Service_Browser(ip, service)

        # Set up sensors
        self.mic = Microphone(sampling, int(time * sampling))

        # Set up protocol and associated processes
        self.protocol_threads = []
        # TODO: Initialize this somewhere where JSON can be read, keep as empty list
        self.protocols = [
            Shurmann_Siggs_Protocol(self.microphone, n, k, timeout, nfs, identity)
        ]

    def start(self):
        print("Starting browser thread.\n")
        self.browser.start_thread()

        print("Starting listening thread.\n")
        output = []
        while 1:
            # Tabs on incoming and outgoing data, as well as exceptions
            readable, writable, exception = select.select(
                self.discoverable, output, self.discoverable
            )

            for incoming in readable:
                # Add if incoming is a new listening socket
                if incoming is self.socket:
                    connection, address = self.socket.accept()
                    connection.setblocking(0)
                    print(f"Connection established with {str(address)}.\n")
                # Read command from established client
                else:
                    data = incoming.recv(8)

                    # Process data or close connection
                    if data:
                        self.service_request(data, incoming)
                    else:
                        self.discoverable.remove(incoming)
                        incoming.close()

            for failed in exception:
                self.discoverable.remove(failed)
                failed.close()

    def service_request(self, data, incoming):
        # Retrieve command, JSON object size, JSON object
        command = data.decode()
        length = int(incoming.revc(4).decode())
        parameters = json.loads(incoming.recv(length).decode())

        # Current device is selected as host
        if command == HOST:
            for protocol in self.protocols:
                # Find the protocol that the message demands
                if protocol.name == parameters.protocol.name:
                    # TODO: Update initialize_protocol to take in JSON
                    participants = self.initialize_protocol(parameters)

                if len(participants) == 0:
                    print("No discoverable devices to perform protocol. Aborting.\n")
                    return False

                # Run the process in the background
                thread = Process(target=protocol.host_protocol(participants))
                thread.start()
                self.protocol_threads.append(thread)
        # Begin protocol
        elif command == STRT:
            for protocol in self.protocols:
                if protocol.name == parameters.protocol.name:
                    thread = Process(target=protocol.device_protocol(incoming))
                    thread.start()

                    # Remove from discoverable as it's running the protocol
                    self.discoverable.remove(incoming)
                    self.protocol_threads.append(thread)

    def initialize_protocol(self, parameters):
        print(f"Initializing {parameters.protocol.name} protocol on all participating devices.")
        bytestream = pickle.dumps(parameters)
        length = len(bytestream).to_bytes(4, byteorder='big')
        message = (STRT.encode() + length + bytestream)
        candidates = self.browser.get_ip_addrs_for_zipa()
        participants = []
        print(f"Potential participants: {str(candidates)}")

        for candidate in candidates:
            # Create a reusable TCP socket through IPv4 broadcasting on the network
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # For devices that couldn't connect to perform the protocol
            failed = connection.connect_ex(candidate, self.port)

            # Send message to begin protocol if connection was successful
            if not failed:
                connection.send(message)
                participants.append(connection)
            else:
                connection.close()
                print(f"Error connecction to {candidate}. Error: {str(failed)}.\n")

        return participants
