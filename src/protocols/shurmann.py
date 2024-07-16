import math
import queue
from typing import Tuple, List, Optional

import numpy as np
from cryptography.hazmat.primitives import constant_time

from networking.network import *
from protocols.protocol_interface import ProtocolInterface


class Shurmann_Siggs_Protocol(ProtocolInterface):
    def __init__(self, parameters: dict, sensor: Any, logger: Any) -> None:
        """
        Implements a signal processing protocol to extract features using Fourier transforms
        and derive cryptographic keys or commitments based on the extracted features.

        :param parameters: Configuration parameters including window length, band length, and other protocol-specific settings.
        :param sensor: Sensor object used to collect data samples.
        :param logger: Logger object for logging protocol operations and data.
        """
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "Shurmann_Siggs_Protocol"
        self.wip = False
        self.window_len = parameters["window_len"]
        self.band_len = parameters["band_len"]
        self.count = 0
        # Conversion from how many requested bits you need to how much sample data you will need for that
        self.time_length = (
            math.ceil(
                (
                    (self.commitment_length * 8)
                    / int((self.window_len / 2 + 1) / self.band_len)
                )
                + 1
            )
            * self.window_len
        )

    def sigs_algo(
        self, x1: List[float], window_len: int = 10000, bands: int = 1000
    ) -> bytes:
        """
        Signal processing algorithm that computes a bit string based on the energy difference between bands of Fourier transforms.

        :param x1: Input signal array.
        :param window_len: Length of the window for Fourier transform.
        :param bands: Number of frequency bands to consider.
        :return: A bit string converted to bytes based on the energy differences.
        """

        def bitstring_to_bytes(s: str) -> bytes:
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        FFTs = []
        from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x)

        x = np.array(x1.copy())
        # wind = scipy.signal.windows.hann(window_len)
        for i in range(0, len(x), window_len):
            if len(x[i : i + window_len]) < window_len:
                # wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i : i + window_len] = x[i : i + window_len]  # * wind
            else:
                x[i : i + window_len] = x[i : i + window_len]  # * wind

            FFTs.append(abs(rfft(x[i : i + window_len])))

        E = {}
        bands_lst = []
        for i in range(0, len(FFTs)):
            frame = FFTs[i]
            bands_lst.append(
                [frame[k : k + bands] for k in range(0, len(frame), bands)]
            )
            for j in range(0, len(bands_lst[i])):
                E[(i, j)] = np.sum(bands_lst[i][j])

        bs = ""

        count = 0
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if E[(i, j)] - E[(i, j + 1)] - (E[(i - 1, j)] - E[(i - 1, j + 1)]) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)

    def zero_out_antialias_sigs_algo(
        self,
        x1: List[float],
        antialias_freq: float,
        sampling_freq: float,
        window_len: int = 10000,
        bands: int = 1000,
    ) -> bytes:
        """
        Similar to sigs_algo but zeroes out frequencies above the anti-aliasing frequency before computing the bit string.

        :param x1: Input signal array.
        :param antialias_freq: Anti-aliasing frequency threshold.
        :param sampling_freq: Sampling frequency of the input signal.
        :param window_len: Length of the window for Fourier transform.
        :param bands: Number of frequency bands to consider.
        :return: A bit string converted to bytes.
        """
        def bitstring_to_bytes(s: str) -> bytes:
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")
        
        FFTs = []
        from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x)

        freq_bin_len = (sampling_freq / 2) / (int(window_len / 2) + 1)
        antialias_bin = int(antialias_freq / freq_bin_len)

        x = np.array(x1.copy())
        # wind = scipy.signal.windows.hann(window_len)
        for i in range(0, len(x), window_len):
            if len(x[i : i + window_len]) < window_len:
                # wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i : i + window_len] = x[i : i + window_len]  # * wind
            else:
                x[i : i + window_len] = x[i : i + window_len]  # * wind

            fft_row = abs(rfft(x[i : i + window_len]))
            FFTs.append(fft_row[:antialias_bin])
        E = {}
        bands_lst = []
        for i in range(0, len(FFTs)):
            frame = FFTs[i]
            bands_lst.append(
                [frame[k : k + bands] for k in range(0, len(frame), bands)]
            )
            for j in range(0, len(bands_lst[i])):
                E[(i, j)] = np.sum(bands_lst[i][j])

        bs = ""
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if (E[(i, j)] - E[(i, j + 1)]) - (
                    E[(i - 1, j)] - E[(i - 1, j + 1)]
                ) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)

    def extract_context(self) -> Tuple[bytes, List[float]]:
        """
        Extracts and processes the signal data from the sensor until the specified sample length is reached.
        
        :return: Tuple containing the processed bits and the raw signal array.
        """
        signal = []
        self.flag.value = 1

        while True:
            try:
                data = self.queue.get()
                signal.extend(data)
            except queue.Empty:
                continue

            if len(signal) >= self.time_length:
                self.flag.value = -1
                break
        # switch anti-aliasing freq to self.sensor.sensor.antialias_sample_rate
        bits = Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(signal, self.sensor.sensor.antialias_sample_rate, self.sensor.sensor.sample_rate, self.window_len, self.band_len)

        #bits = self.sigs_algo(signal, window_len=self.window_len, bands=self.band_len)
        return bits, signal

    def parameters(self, is_host: bool) -> str:
        """
        Returns a formatted string of protocol parameters for logging purposes.

        :param is_host: Boolean indicating if the host's parameters are to be returned.
        :return: Formatted string of parameters.
        """
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"window_length: {self.window_len}\n"
        parameters += f"band_length: {self.band_len}\n"
        parameters += f"time_length: {self.time_length}\n"

    def device_protocol(self, host: socket.socket) -> None:
        """
        Executes the device side protocol which involves sending and receiving data to/from the host.

        :param host: Socket connection to the host.
        """
        host.setblocking(1)
        if self.verbose:
            print(f"Iteration {str(self.count)}.\n")

        # Log parameters to NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("Sending ACK.\n")
        ack(host)

        # Wait for ack from host to begin context extract, quit early if no response within time
        if self.verbose:
            print("Waiting for ACK from host.")
        if not ack_standby(host, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return

        # Extract bits from mic
        if self.verbose:
            print("Extracting context\n")
        witness, signal = self.extract_context()

        if self.verbose:
            print("witness: " + str(witness))

        # Wait for Commitment
        if self.verbose:
            print("Waiting for commitment from host")
        commitments, recieved_hashes = commit_standby(host, self.timeout)

        commitment = commitments[0]
        recieved_hash = recieved_hashes[0]

        # Early exist if no commitment recieved in time
        if not commitment:
            if self.verbose:
                print("No commitment recieved within time limit - early exit\n")
            return

        # Decommit
        if self.verbose:
            print("Decommiting")
        key = self.re.decommit_witness(commitment, witness)

        generated_hash = self.hash_function(key)

        success = False
        if constant_time.bytes_eq(generated_hash, recieved_hash):
            success = True

        if self.verbose:
            print(f"key: {str(key)}\n success: {str(success)}\n")

        self.logger.log(
            [
                ("witness", "txt", witness),
                ("commitment", "txt", commitment),
                ("success", "txt", str(success)),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
        )

        self.count += 1

    def host_protocol_single_threaded(self, device_socket: socket.socket) -> None:
        """
        Manages the protocol operations for a single device connection in a threaded environment.

        :param device_socket: The socket connection to a single device.
        """
        # Exit early if no devices to pair with
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return
        if self.verbose:
            print("Successfully ACKed participating device")
            print()

        if self.verbose:
            print("ACKing all participating devices")
        ack(device_socket)

        # Extract key from mic
        if self.verbose:
            print("Extracting Context\n")
        witness, signal = self.extract_context()

        # Commit Secret
        if self.verbose:
            print("Commiting Witness")

        secret_key, commitment = self.re.commit_witness(witness)

        if self.verbose:
            print("witness: " + str(witness))

        hash = self.hash_function(secret_key)

        if self.verbose:
            print("Sending commitment")
            print()
        send_commit([commitment], [hash], device_socket)

        self.logger.log(
            [
                ("witness", "txt", str(witness)),
                ("commitment", "txt", commitment),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
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
    prot = Shurmann_Siggs_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024)),
                                   8,
                                   4,
                                   10000,
                                   1000,
                                   10,
                                   None,
    )
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()"""
