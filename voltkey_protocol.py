import multiprocessing as mp

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes

from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj

OUTLET_FREQ = 60  # Hz


class VoltKeyProtocol:
    def __init__(
        self,
        sensor,
        key_length,
        parity_symbols,
        periods,
        bins,
        timeout,
        logger,
        verbose=True,
    ):
        # General protocol information
        self.name = "voltkey"
        self.sensor = sensor
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose
        self.hash = hashes.SHA256()
        self.count = 0

        # Protocol specific information
        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        # TODO parameters will send this
        self.periods = periods + 1  # + 1 Compensating for dropped frames
        self.bins = bins
        self.host = False
        self.time_length = int((1 / OUTLET_FREQ) * periods * 8) * 100
        self.period_length = None
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )

    def extract_signal(self):
        signal = self.sensor.read(self.time_length)
        # print(f"From extract_signal(). Host? {self.host}: {signal}")
        return signal

    def extract_context(self, filtered_signal):
        """
        VoltKey protocol host needs preamble from device. Cannot perform bit
        extraction until host dataset is synchronized with device dataset
        """

        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        bits = self.voltkey_algo(filtered_signal)

        return bitstring_to_bytes(bits)

    def sync(self, signal, preamble):
        synced_frames = None

        # Focusing on one period with buffer space
        for i in range(
            len(signal) // self.periods + ((len(signal) // self.periods) // 2)
        ):
            synced = True

            # Evaluate if shift aligns frames with preamble
            for j in range(len(preamble)):
                if preamble[j] != signal[i + j]:
                    synced = False
                    break

            # Adjust frames
            if synced == True:
                synced_frames = signal[i:]
                break

        return synced_frames

    def signal_processing(self, signal):
        """
        Assuming the length of a sinusoid period

        Working with values ranging from 1-48600
        """
        filtered_frames = []
        # Iterate through each period; preamble is skipped due to public knowledge
        for period in range(self.period_length, len(signal) - self.period_length, self.period_length):
            # print(f"Host? {self.host} // {len(signal)} // {period}")
            current = signal[period : period + self.period_length]
            next = signal[period + self.period_length : period + 2 * self.period_length]
            # print(f"In signal processing. Host? {self.host}\nNext: {next}")
            result = []
            # TODO add conditional that will handle if last cycle isn't filled up. Skip out on it
            # Substracts sinusoid wave, leaving any existing noise
            if len(current) == len(next):
                for point in range(len(current)):
                    noise = current[point] - next[point]
                    result.append(noise)
                    # print(f"Result. Host? {self.host}\n{result}")

                filtered_frames.extend(result)
            

        return filtered_frames

    def gen_key(self, signal):
        bits = ""
        # TODO rewrite to use discovered period length
        step = len(signal) // (
            (self.periods - 1) * self.bins
        )  # - 1 as preamble's nuked
        # print(f"Host? {self.host}. Step: {step}. Signal: {len(signal)}\nPeriods: {self.periods}. Bins: {self.bins}")

        for i in range(0, len(signal), step):
            # Working in bins, average value of each bin will act as baseline
            current_bin = signal[i : i + step]
            average = deviation = sum(current_bin) / len(current_bin)

            for data in current_bin:
                # Calculate absolute value of sample relative to the average value
                absolute_value = average + abs(data - average)

                # Preserve original value, used to determine `1` or `0`
                if absolute_value > deviation:
                    deviation = data

            bits += "1" if deviation > average else "0"

        return bits
    
    def find_period_length(self, signal):
        """
        TODO Consult voltkey code to improve this

        To be used on both devices. Client must use first to be able to share preamble period
        for frame synchronization. Host may use after shaving
        frames.
        """
        length = 0
        start = signal[0]
        # Next cycle will follow same direction; that's when we're in a new period
        next = signal[1]
        #print(f"Sample of signal in find period length. Host? {self.host}: \n{signal[0:10]}")
        # print(f"start: {start}, next: {next}\nLength of signal: {len(signal)}")
        
        for point in range(len(signal) - 1):
            # print(signal[point])
            # Still within cycle
            if signal[point] != start and signal[point + 1] != next:
                length += 1
            # Reached the end of a sin cycle
            elif signal[point] == start and signal[point + 1] == next and point != 0:
                break

        print(f"Length approximately: {length}")
        return length

    def voltkey_algo(self, signal):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        # Assumes that host has synced frames from client preamble
        filtered = self.signal_processing(signal)
        # Once processed, make further chunks to be used to create bits
        # print(f"Host? {self.host} Filtered frames: {filtered}\n")
        key = self.gen_key(filtered)

        return bitstring_to_bytes(key)

    def device_protocol(self, host):
        host.setblocking(1)

        if self.verbose:
            print(f"Iteration: {self.count}.\n")

        # self.logger.log([("parameters", "txt", self.parameters(False))])

        if self.verbose:
            print("Sending ACK.\n")
        ack(host)

        if self.verbose:
            print("Waiting for host ACK.\n")

        # Abort if ACK not recieved on time
        if not ack_standby(host, self.timeout):
            if self.verbose:
                print("Timed out: host ACK not recieved.\n")

            return

        if self.verbose:
            print("Extracting context.\n")
        signal = self.extract_signal()
        # print(f"Signal collected from Device. Sample: {signal}")
        self.period_length = self.find_period_length(signal) # Won't ever see a 0 need to adjust
        time.sleep(5)
        # Get first period, send to host for data synchronization
        preamble = signal[:self.period_length]
        # print(f"In client. Sending preamble: {preamble}")
        send_preamble(host, preamble)

        # Bit extraction sequence
        witness = self.voltkey_algo(signal)

        if self.verbose:
            print(f"Witness: '{witness}'.\n")

        if self.verbose:
            print("Waiting for host commitment.\n")
        commitment, recieved_hash = commit_standby(host, self.timeout)

        # Abort if commitment's not recieved on time
        if not commitment:
            if self.verbose:
                print("Timed out: host commitment not recieved.\n")

            return

        if self.verbose:
            print("Decommitting.\n")
        key = self.re.decommit_witness(commitment, witness)
        generated_hash = self.hash_function(key)

        success = (
            True if constant_time.bytes_eq(recieved_hash, generated_hash) else False
        )

        if self.verbose:
            print(f"Generated key: {key}.\n\nSuccess: {success}.\n")

        """
        self.logger.log(
            [
                ("witness", "txt", witness),
                ("commitment", "txt", commitment),
                ("success", "txt", str(success)),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
        )
        """

        self.count += 1

    def host_protocol(self, devices):

        # self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print(f"Iteration {self.count}.\n")

        for device in devices:
            process = mp.Process(target=self.host_protocol_threaded, args=[device])
            process.start()

    def host_protocol_threaded(self, device):
        self.host = True

        if not ack_standby(device, self.timeout):
            if self.verbose:
                print("Timed out: No client devices ACKed.\n")

            return
        else:
            print("Successfully ACKed parcitipating client devices.\n")
            print("ACKing all participating client devices.\n")
        ack(device)

        if self.verbose:
            print("Extracting context.\n")
        signal = self.extract_signal()
        # print(f"Signal collected from Host. Sample: {signal}")

        # Synchronize frames with client's preamble
        if self.verbose:
            print("Waiting for client preamble.\n")
        preamble = get_preamble(device, self.timeout)
        # print(f"In host protocol. Preamble: {preamble}")

        if not preamble:
            if self.verbose:
                print("Timed out: Client preamble not recieved.\n")
            
            return

        # Deriving period length from preamble
        # OPTION: Host gets period length from preamble, or can figure out its own
        self.period_length = len(preamble)
        print(f"Host has preamble length of {len(preamble)}")

        # Drop frames according to preamble
        synced_frames = self.sync(signal, preamble)
        # With signal aligned, host begins voltkey protocol
        witness = self.voltkey_algo(synced_frames)

        if self.verbose:
            print(f"Witness: '{witness}'.\n")

        if self.verbose:
            print("Committing witness.\n")
        secret_key, commitment = self.re.commit_witness(witness)
        generated_hash = self.hash_function(secret_key)

        if self.verbose:
            print("Sending commitment.\n")
        send_commit(commitment, generated_hash, device)

        """
        self.logger.log(
            [
                ("witness", "txt", str(witness)),
                ("commitment", "txt", commitment),
                ("signal", "csv", ", ".join(str(num) for num in signal)),
            ]
        )
        """

        self.count += 1

    def parameters(self, is_host):
        """
        Converts parameters from JSON to string format for NFS logging
        """
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"window_length: {self.periods - 1}\n"
        parameters += f"band_length: {self.bins}\n"
        parameters += f"time_length: {self.time_length}\n"

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash)
        hash_func.update(bytes)

        return hash_func.finalize()


# Test code
# TODO Don't forget to uncomment logger code
import socket


def device(protocol):
    print("DEVICE")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))

    protocol.device_protocol(s)


def host(protocol):
    print("HOST")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()

    conn, addr = s.accept()
    s.setblocking(0)

    protocol.host_protocol([conn])


if __name__ == "__main__":
    from test_sensor import Test_Sensor
    from sensor_reader import Sensor_Reader

    print("Testing VoltKey protocol.\n")

    sample_rate = 25
    ts = Test_Sensor(sample_rate, sample_rate * 17, sample_rate * 4)
    sr = Sensor_Reader(ts)
    protocol = VoltKeyProtocol(sr, 8, 4, 16, 8, 10, None, True)
    time.sleep(3)
    host_process = mp.Process(target=host, args=[protocol])
    device_process = mp.Process(target=device, args=[protocol])
    # time.sleep(3)
    host_process.start()
    device_process.start()
