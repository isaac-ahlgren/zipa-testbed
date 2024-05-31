import json
import os
import select
import socket
from multiprocessing import Process

import netifaces as ni
import yaml

from networking.browser import ZIPA_Service_Browser
from networking.network import *
from networking.nfs import NFSLogger
from protocols.miettinen import Miettinen_Protocol
from protocols.perceptio import Perceptio_Protocol
from protocols.shurmann import Shurmann_Siggs_Protocol
from protocols.voltkey_protocol import VoltKeyProtocol
from sensors.sensor_collector import Sensor_Collector
from sensors.sensor_reader import Sensor_Reader

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
                if protocol.name == parameters["protocol"]["name"]:
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
        # Begin protocol
        elif command == STRT:
            print("Device selected as a client.")

            for protocol in self.protocols:
                if protocol.name == parameters["protocol"]["name"]:
                    thread = Process(target=protocol.device_protocol, args=[incoming])
                    thread.start()

                    # Remove from discoverable as it's running the protocol
                    self.discoverable.remove(incoming)
                    self.protocol_threads.append(thread)

    def initialize_protocol(self, parameters):
        print(
            f"Initializing {parameters['protocol']['name']} protocol on all participating devices."
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

    def create_protocol(self, parameters):
        name = parameters["protocol"]["name"]

        if parameters["sensor"] not in self.sensors:
            print("Sensor not supported")
            return

        match name:
            case "shurmann-siggs":
                self.protocols.append(
                    Shurmann_Siggs_Protocol(
                        self.sensors[parameters["sensor"]],
                        parameters["key_length"],
                        parameters["parity_symbols"],
                        parameters["protocol"]["window_len"],
                        parameters["protocol"]["band_len"],
                        parameters["timeout"],
                        self.logger,
                    )
                )

            case "miettinen":
                self.protocols.append(
                    Miettinen_Protocol(
                        self.sensors[parameters["sensor"]],
                        parameters["key_length"],
                        parameters["parity_symbols"],
                        parameters["protocol"]["f"],
                        parameters["protocol"]["w"],
                        parameters["protocol"]["rel_thresh"],
                        parameters["protocol"]["abs_thresh"],
                        parameters["protocol"]["auth_thresh"],
                        parameters["protocol"]["success_thresh"],
                        parameters["protocol"]["max_iterations"],
                        parameters["timeout"],
                        self.logger,
                    )
                )

            case "voltkey":
                self.protocols.append(
                    VoltKeyProtocol(
                        self.sensors[parameters["sensor"]],
                        parameters["key_length"],
                        parameters["parity_symbols"],
                        parameters["protocol"]["periods"],
                        parameters["protocol"]["bins"],
                        parameters["timeout"],
                        self.logger,
                    )
                )

            case "perceptio":
                self.protocols.append(
                    Perceptio_Protocol(
                        self.sensors[parameters["sensor"]],
                        parameters["key_length"],
                        parameters["parity_symbols"],
                        parameters["time_length"],
                        parameters["protocol"]["a"],
                        parameters["protocol"]["cluster_sizes_to_check"],
                        parameters["protocol"]["cluster_th"],
                        parameters["protocol"]["top_th"],
                        parameters["protocol"]["bottom_th"],
                        parameters["protocol"]["lump_th"],
                        parameters["protocol"]["conf_thresh"],
                        parameters["protocol"]["max_iterations"],
                        parameters["protocol"]["sleep_time"],
                        parameters["protocol"]["max_no_events_detected"],
                        parameters["timeout"],
                        self.logger,
                    )
                )
            case _:
                print("Protocol not supported.")

    def get_sensor_configs(self, yaml_file):
        with open(f"{yaml_file}", "r") as f:
            config_params = yaml.safe_load(f)
        time_to_collect = config_params["time_collected"]
        sensors_used = config_params["sensors_used"]
        sensor_sample_rates = config_params["sensor_sample_rates"]
        chunk_sizes = config_params["chunk_sizes"]
        return time_to_collect, sensors_used, sensor_sample_rates, chunk_sizes

    # TODO: make sht31d an option as a sensor
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

        if sensors_used["microphone"]:
            from sensors.microphone import Microphone

            self.devices["microphone"] = Microphone(
                sample_rates["microphone"],
                sample_rates["microphone"] * time_length,
                chunk_sizes["microphone"],
            )

        if sensors_used["bmp280"]:
            from sensors.bmp280 import BMP280Sensor

            self.devices["bmp280"] = BMP280Sensor(
                sample_rates["bmp280"],
                sample_rates["bmp280"] * time_length,
                chunk_sizes["bmp280"],
            )

        if sensors_used["pir"]:
            from sensors.pir import PIRSensor

            self.devices["pir"] = PIRSensor(
                sample_rates["pir"],
                sample_rates["pir"] * time_length,
                chunk_sizes["pir"],
            )

        if sensors_used["veml7700"]:
            from sensors.veml7700 import LightSensor

            self.devices["veml7700"] = LightSensor(
                sample_rates["veml7700"],
                sample_rates["veml7700"] * time_length,
                chunk_sizes["veml7700"],
            )

        if sensors_used["sht31d"]:
            from sensors.sht31d import HumiditySensor

            self.devices["sht31d"] = HumiditySensor(
                sample_rates["sht31d"],
                sample_rates["sht31d"] * time_length,
                chunk_sizes["sht31d"],
            )

        if sensors_used["voltkey"]:
            from sensors.voltkey import Voltkey

            self.devices["voltkey"] = Voltkey(
                sample_rates["voltkey"],
                sample_rates["voltkey"] * time_length,
                chunk_sizes["voltkey"],
            )

        if sensors_used["test_sensor"]:
            from sensors.test_sensor import Test_Sensor

            self.devices["test_sensor"] = Test_Sensor(
                sample_rates["test_sensor"],
                sample_rates["test_sensor"] * time_length,
                chunk_sizes["test_sensor"],
            )

        # Wrap physical sensors into respective sensor wrapper
        if collection_mode:
            for device in self.devices:
                self.sensors[device] = Sensor_Collector(
                    self.devices[device], self.logger
                )
        else:
            for device in self.devices:
                self.sensors[device] = Sensor_Reader(self.devices[device])
