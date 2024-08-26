from typing import Any, List

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from networking.network import (
    ack,
    ack_standby,
    commit_standby,
    get_nonce_msg_standby,
    send_commit,
    socket,
)
from protocols.common_protocols import (
    diffie_hellman,
    send_nonce_msg_to_device,
    send_nonce_msg_to_host,
    verify_mac_from_device,
    verify_mac_from_host,
)
from protocols.protocol_interface import ProtocolInterface
from signal_processing.miettinen import MiettinenProcessing
from error_correction.corrector import Fuzzy_Commitment
from error_correction.reed_solomon import ReedSolomonObj


class Miettinen_Protocol(ProtocolInterface):
    def __init__(self, parameters: dict, sensor: Any, logger: Any):
        """
        Initializes a new instance of the Miettinen Protocol with specified parameters for key generation and communication.

        :param sensor: The sensor object that provides access to real-time data.
        :param logger: A logging object used to record protocol activity and debugging information.
        :param parameters: A dictionary filled with the parameters needed to initialize and run the protocol
        """
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.f = parameters["f"] * self.sensor.sensor.sample_rate
        self.w = parameters["w"] * self.sensor.sensor.sample_rate
        self.key_length = parameters["key_length"]
        self.parity_symbols = parameters["parity_symbols"]
        self.commitment_length = self.key_length + self.parity_symbols
        self.rel_thresh = parameters["rel_thresh"]
        self.abs_thresh = parameters["abs_thresh"]
        self.auth_threshold = parameters["auth_thresh"]
        self.success_threshold = parameters["success_thresh"]
        self.max_iterations = parameters["max_iterations"]
        self.timeout = parameters["timeout"]
        self.name = "Miettinen_Protocol"
        self.wip = False
        self.ec_curve = ec.SECP384R1()
        self.nonce_byte_size = 16
        self.time_length = (self.w + self.f) * (self.commitment_length * 8 + 1)
        self.count = 0
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )

    def process_context(self) -> List[bytes]:
        """
        Processes the captured context to detect signals and generate binary representations.

        This method reads samples using a predefined time length, resets flags, clears any existing queues,
        and applies a specific algorithm to process the signal into a binary format suitable for cryptographic or other forms of processing.

        :return: A list containing a single bytes object which is the binary representation of the processed signal.
        """
        # TODO: Signal must be logged somehow
        signal = self.read_samples(self.time_length)

        # Taken from read_samples in protocol_interface
        ProtocolInterface.reset_flag(self.queue_flag)
        self.clear_queue()

        bits = MiettinenProcessing.miettinen_algo(
            signal, self.f, self.w, self.rel_thresh, self.abs_thresh
        )

        return [bits]

    def parameters(self, is_host: bool) -> str:
        """
        Constructs a string detailing the current settings and parameters of the protocol.

        :param is_host: Boolean indicating whether the device is host.
        :return: A string containing formatted protocol parameters.
        """
        parameters = ""
        parameters += "protocol: " + self.name
        parameters += "is_host: " + str(is_host)
        parameters += (
            "\nsensor: " + self.sensor.sensor.name
        )  # Name's in physical sensor.
        parameters += "\nkey_length: " + str(self.key_length)
        parameters += "\nparity_symbols: " + str(self.parity_symbols)
        parameters += "\nf: " + str(self.f)
        parameters += "\nw: " + str(self.w)
        parameters += "\nrel_thresh: " + str(self.rel_thresh)
        parameters += "\nabs_thresh: " + str(self.abs_thresh)
        parameters += "\nsuccess_threshold: " + str(self.success_threshold)
        parameters += "\nauthentication_threshold: " + str(self.auth_threshold)
        parameters += "\nmax_iterations: " + str(self.max_iterations)
        parameters += "\ntime_length: " + str(self.time_length)
        return parameters

    def device_protocol(self, host_socket: socket.socket) -> None:
        """
        Implements the device side of the protocol, handling communications and key exchange operations.

        :param host_socket: The socket connected to the host.
        """
        host_socket.setblocking(1)

        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("\nSending ACK")
        ack(host_socket)

        # Wait for ack from host to being context extract, quit early if no response within time
        if self.verbose:
            print("\nWaiting for ACK from host")

        if not ack_standby(host_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit\n")
            return

        # Shared key generated
        shared_key = diffie_hellman(
            host_socket, self.ec_curve, self.timeout, self.verbose
        )

        current_key = shared_key
        successes = 0
        total_iterations = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            # Sending ack that they are ready to begin

            if self.verbose:
                print("Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Sending ACK\n")
            ack(host_socket)

            success = False

            if self.verbose:
                print("[HOST] Extracting context.")
            # Extract bits from sensor
            witness = self.get_context()
            witness = witness[0]

            # Wait for Commitment
            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)

            commitment = commitments[0]

            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return

            if self.verbose:
                print("witness: " + str(witness))

            # Decommit

            if self.verbose:
                print("Decommiting")
            prederived_key = self.re.decommit_witness(commitment, witness)

            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Send nonce message to host
            generated_nonce = send_nonce_msg_to_host(
                host_socket,
                pd_key_hash,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                return

            # If hashes are equal, then it was successful
            if verify_mac_from_host(
                recieved_nonce_msg,
                generated_nonce,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            ):
                success = True
                successes += 1
                current_key = derived_key

            if self.verbose:
                print("Produced Key: " + str(derived_key))
                print(
                    f"success: f{str(success)}, Number of successes: f{str(successes)}, Total number of iterations: {str(total_iterations)}"
                )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    # ("signal", "csv", signal),
                ],
                count=total_iterations,
            )

            # Increment total number of iterations key evolution has occured
            total_iterations += 1

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    f"Total Key Pairing Success: auth - {str(successes / total_iterations)}"
                )
            else:
                print(
                    f"Total Key Pairing Failure: auth - {str(successes / total_iterations)}"
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {str(successes)} total_iterations: {str(total_iterations)} succeeded: {str(successes / total_iterations >= self.auth_threshold)}",
                )
            ]
        )

        self.count += 1

    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Handles protocol operations for a single device from the host's perspective in a separate thread.

        :param device_socket: The socket for communication with a specific device.
        """
        device_socket.setblocking(1)

        device_ip_addr, device_port = device_socket.getpeername()

        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return

        if self.verbose:
            print("ACKing participating device")
        ack(device_socket)

        # Shared key generated
        shared_key = diffie_hellman(
            device_socket, self.ec_curve, self.timeout, self.verbose
        )

        current_key = shared_key
        total_iterations = 0
        successes = 0
        while (
            successes < self.success_threshold
            and total_iterations < self.max_iterations
        ):
            success = False
            # ACK device
            if self.verbose:
                ack(device_socket)

            if not ack_standby(device_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Successfully ACKed participating device\n")

            # Extract key from sensor
            if self.verbose:
                print("[CLIENT] Extracting context.")
            witness = self.get_context()
            witness = witness[0]

            # Commit Secret
            if self.verbose:
                print("Commiting Witness")
            prederived_key, commitment = self.re.commit_witness(witness)

            if self.verbose:
                print("witness: " + str(witness))
                print()

            if self.verbose:
                print("Sending commitment")
                print()

            # Hash prederived key
            pd_key_hash = self.hash_function(prederived_key)

            # Key Confirmation Phase
            send_commit([commitment], [pd_key_hash], device_socket)

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                    print()
                return

            # Derive new key using previous key and new prederived key from fuzzy commitment
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(prederived_key + current_key)

            if verify_mac_from_device(
                recieved_nonce_msg,
                derived_key,
                pd_key_hash,
                self.nonce_byte_size,
                self.hash_func,
            ):
                success = True
                successes += 1
                current_key = derived_key

            # Create and send key confirmation value
            send_nonce_msg_to_device(
                device_socket,
                recieved_nonce_msg,
                derived_key,
                pd_key_hash,
                self.nonce_byte_size,
                self.hash_func,
            )

            self.logger.log(
                [
                    ("witness", "txt", witness),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    # ("signal", "csv", signal),
                ],
                count=total_iterations,
                ip_addr=device_ip_addr,
            )

            # Increment total times key evolution has occured
            total_iterations += 1

            if self.verbose:
                print(
                    f"success: {str(success)}, Number of successes: {str(successes)}, Total number of iterations: {str(total_iterations)}"
                )
                print()

        if self.verbose:
            if successes / total_iterations >= self.auth_threshold:
                print(
                    f"Total Key Pairing Success: auth - {str(successes / total_iterations)}"
                )
            else:
                print(
                    f"Total Key Pairing Failure: auth - {str(successes / total_iterations)}"
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {str(successes)} total_iterations: {str(total_iterations)} succeeded: {str(successes / total_iterations >= self.auth_threshold)}",
                )
            ],
            ip_addr=device_ip_addr,
        )

        self.count += 1


"""###TESTING CODE###
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
    from test_sensor import Test_Sensor
    from sensor_reader import Sensor_Reader
    prot = Miettinen_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024)),
        8,
        4,
        1,
        1,
        0.003,
        2.97e-05,
        0.9,
        5,
        20,
        10)
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()"""
