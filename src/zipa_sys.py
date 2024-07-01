import json
import os
import select
import socket
import pkgutil
import inspect
from multiprocessing import Process

import netifaces as ni
import yaml

from networking.browser import ZIPA_Service_Browser
from networking.network import *
from networking.nfs import NFSLogger
from protocols.protocol_interface import ProtocolInterface
from sensors.sensor_collector import Sensor_Collector
from sensors.sensor_interface import SensorInterface
from sensors.sensor_reader import Sensor_Reader
import protocols
import sensors

# Used to initiate and begin protocol
HOST = "host    "
STRT = "start   "


class ZIPA_System:
    def __init__(
        self,
        identity,
        service,
        nfs_dir,
        collection_mode=False,
        only_locally_store=False,
    ):

        self.collection_mode = collection_mode

        # Create data directory if it does not already exist
        if not os.path.exists("./local_data"):
            os.mkdir("./local_data")

        # Set up Logger
        self.id = identity
        self.nfs_dir = nfs_dir
        self.logger = NFSLogger(
            user="USERNAME",
            password="PASSWORD",
            host="SERVER IP",
            database="file_log",
            nfs_server_dir=self.nfs_dir,  # Make sure this directory exists and is writable
            local_dir="./local_data/",
            identifier="DEVICE IDENTIFIER",  # Could be IP address or any unique identifier
            use_local_dir=only_locally_store,
        )

        # Set up sensors
        time_to_collect, sensors_used, sample_rates, chunk_sizes = (
            self.get_sensor_configs(os.getcwd() + "/src/sensors/sensor_config.yaml")
        )
        self.create_sensors(
            time_to_collect,
            sensors_used,
            sample_rates,
            chunk_sizes,
            collection_mode=collection_mode,
        )

        # Set up infrastructure if not being run in collection mode
        if not collection_mode:
            # Set up a listening socket
            self.ip = ni.ifaddresses("wlan0")[ni.AF_INET][0]["addr"]
            self.port = 5005

            # Create a reusable TCP socket through IPv4 broadcasting on the network
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind socket to the specified socket and IPv4 address that runs continuously
            self.socket.bind((self.ip, self.port))
            self.socket.setblocking(0)
            self.socket.listen()

            # Set up a discovery for the protocol
            self.discoverable = [self.socket]
            self.browser = ZIPA_Service_Browser(self.ip, service)

            # Set up protocol and associated processes
            self.protocol_threads = []
            self.protocols = []

    def start(self):

        # If in collection mode, no need to handle any incoming network connections
        if self.collection_mode:
            return

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
                    self.discoverable.append(connection)
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
        length = int.from_bytes(incoming.recv(4), byteorder="big")
        parameters = json.loads(incoming.recv(length))
        # Create protocol based on JSON
        self.create_protocol(parameters)

        print("Determining device role.")
        # Current device is selected as host
        if command == HOST:
            print("Device selected as the host.")

            for protocol in self.protocols:
                # Find the protocol that the message demands
                if protocol.name == parameters["name"] and not protocol.wip:
                    participants = self.initialize_protocol(parameters)

                    if len(participants) == 0:
                        print(
                            "No discoverable devices to perform protocol. Aborting.\n"
                        )
                        return False

                    # Run the process in the background
                    thread = Process(target=protocol.host_protocol, args=[participants])
                    thread.start()
                    self.protocol_threads.append(thread)

                else:
                    print(
                        f"Requested protocol is not ready for use. Skipping {protocol.name}.\n"
                    )

        # Begin protocol
        elif command == STRT:
            print("Device selected as a client.")

            for protocol in self.protocols:
                if protocol.name == parameters["name"] and not protocol.wip:
                    thread = Process(target=protocol.device_protocol, args=[incoming])
                    thread.start()

                    # Remove from discoverable as it's running the protocol
                    self.discoverable.remove(incoming)
                    self.protocol_threads.append(thread)
                else:
                    print(
                        f"Requested protocol is not ready for use. Skipping {protocol.name}.\n"
                    )

    def initialize_protocol(self, parameters):
        print(
            f"Initializing {parameters['name']} protocol on all participating devices."
        )
        bytestream = json.dumps(parameters).encode("utf8")
        length = len(bytestream).to_bytes(4, byteorder="big")
        message = STRT.encode() + length + bytestream
        candidates = self.browser.get_ip_addrs_for_zipa()
        participants = []
        print(f"Potential participants: {str(candidates)}")

        for candidate in candidates:
            # Create a reusable TCP socket through IPv4 broadcasting on the network
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # For devices that couldn't connect to perform the protocol
            failed = connection.connect_ex((candidate, self.port))

            # Send message to begin protocol if connection was successful
            if not failed:
                connection.send(message)
                participants.append(connection)
            else:
                connection.close()
                print(f"Error connecction to {candidate}. Error: {str(failed)}.\n")

        return participants

    def create_protocol(self, payload):
        requested_name = payload["name"]
        sensor = payload["parameters"]["sensor"]

        if sensor not in self.sensors:
            raise Exception("Sensor not supported")

        for _, module_name, _ in pkgutil.iter_modules(protocols.__path__):
            module = __import__(f"protocols.{module_name}", fromlist=module_name)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, ProtocolInterface)
                    and obj is not ProtocolInterface
                    and name == requested_name
                ):
                    self.protocols.append(
                        obj(payload["parameters"], self.sensors[sensor], self.logger)
                    )
                    break

    def get_sensor_configs(self, yaml_file):
        with open(f"{yaml_file}", "r") as f:
            config_params = yaml.safe_load(f)
        time_to_collect = config_params["time_collected"]
        sensors_used = config_params["sensors_used"]
        sensor_sample_rates = config_params["sensor_sample_rates"]
        chunk_sizes = config_params["chunk_sizes"]
        return time_to_collect, sensors_used, sensor_sample_rates, chunk_sizes

    def create_sensors(
        self,
        time_length,
        sensors_used,
        sample_rates,
        chunk_sizes,
        collection_mode=False,
    ):
        # Create instances of physical sensors
        self.devices = {}
        self.sensors = {}

        for _, module_name, _ in pkgutil.iter_modules(sensors.__path__):
            module = __import__(f"sensors.{module_name}", fromlist=module_name)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, SensorInterface)
                    and obj is not SensorInterface
                    and sensors_used[name]
                ):
                    self.devices[name] = obj(
                        sample_rates[name],
                        sample_rates[name] * time_length,
                        chunk_sizes[name],
                    )

        if collection_mode:
            for device in self.devices:
                self.sensors[device] = Sensor_Collector(
                    self.devices[device], self.logger
                )
        else:
            for device in self.devices:
                self.sensors[device] = Sensor_Reader(self.devices[device])
