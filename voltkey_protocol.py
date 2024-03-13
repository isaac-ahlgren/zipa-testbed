import multiprocessing as mp

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes

from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj


class Voltkey_Protocol:
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
        self.name = "voltkey"
        self.sensor = sensor
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        # TODO parameters will send this
        self.periods = periods + 1  # + 1 Compensating for dropped frames
        self.bins = bins
        self.host = False
        self.time_length = (
            self.commitment_length * 8 + 1
        )  # TODO redo with periods, bins in mind
        self.period_length = None
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, self.key_length), self.key_length
        )
        self.hash = hashes.SHA256()

    def extract_signal(self):
        return self.sensor.read(self.time_length)

    def extract_context(self, filtered_signal):
        """
        VoltKey protocol host needs preamble from device. Cannot perform bit
        extraction until host dataset is synchronized with device dataset
        """

        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        bits = self.voltkey_algo(filtered_signal)

        return bitstring_to_bytes(bits)

    # TODO align sinusoidal waves before signal processing
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

    def signal_processing(self, signal, period_length):
        """
        Assuming the length of a sinusoid period

        Working with values ranging from 1-48600
        """
        filtered_frames = []
        # Iterate through each period; preamble is skipped due to public knowledge
        for period in range(period_length, len(signal) - period_length, period_length):
            current = signal[period : period + period_length]
            next = signal[period + period_length : period + 2 * period_length]
            result = []

            # Substracts sinusoid wave, leaving any existing noise
            for point in range(len(current)):
                result[point] = current[point] - next[point]

            filtered_frames.append(result)

        return filtered_frames

    def gen_key(self, signal):
        bits = ""
        step = len(signal) // (
            (self.periods - 1) * self.bins
        )  # - 1 as preamble's nuked

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
        TODO Consult volkey code to improve this

        To be used on both devices. Client must use first to be able to share preamble period
        for frame synchronization. Host may use after shaving
        frames.
        """
        length = start = None

        for i in signal:
            # Begin counting frames in period
            if start == None and length == None:  # TODO improve naive approach
                start = i
                length = 0
            # Still within sin period
            elif signal[i] != i and length >= 0:
                length += 1
            # Reached the end of a sin period
            elif signal[i] == i and length >= 0:
                break

        return length

    def voltkey_algo(self, signal):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        # Assumes that host has synced frames from client preamble
        filtered = self.signal_processing(signal)
        # Once processed, make further chunks to be used to create bits based on
        # commitment length
        key = self.gen_key(filtered)

        return bitstring_to_bytes(key)

    def device_protocol(self, host):
        # Consult networking code of other protocols
        # Host and client acknowledge each other and begin
        # Host and client extract sensor data
        # Client finds period length
        # Client gets subset of data known as preamble
        # Client sends preamble to host
        # Client begins voltkey protocol
        pass

    def host_protocol(self, devices):
        pass

    def host_protocol_threaded(self, devices):
        self.host = True
        # Host and client acknowledge each other and begin
        # Host and client extract sensor data
        # Host waits for client's preamble
        # Host drops frames according to preamble
        # OPTION: Host gets period length from preamble, or can figure out its own
        # With signal aligned, host begins voltkey protocol
        pass
