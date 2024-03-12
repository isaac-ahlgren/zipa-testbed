import multiprocessing as mp

import numpy as np
from cryptography.hazmat.primitives import constant_time, hashes

from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj


class Voltkey_Protocol:
    def __init__(self, sensor, key_length, parity_symbols, periods, bins, timeout, logger, verbose=True):
        self.name = "voltkey"
        self.sensor = sensor
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        # TODO parameters will send this
        self.periods = periods + 1 # + 1 Compensating for dropped frames
        self.bins = bins
        self.time_length = (self.commitment_length * 8 + 1) # TODO redo with periods, bins in mind
        self.re = Fuzzy_Commitment(ReedSolomonObj(self.commitment_length, self.key_length), self.key_length)
        self.hash = hashes.SHA256()


    def extract_context(self):
        # Getting short ints
        signal = self.sensor.read(self.time_length)
        bits = self.voltkey_algo(signal)

        return bits, signal
    
    # TODO align sinusoidal waves before signal processing
    def sync(self, host_frames, client_preamble):
        synced_frames = None

        
        # Focusing on one period with buffer space
        for i in range(len(host_frames) // self.periods + ((len(host_frames) // self.periods) // 2)):
            synced = True

            # Evaluate if shift aligns frames with preamble
            for j in range(len(client_preamble)):
                if client_preamble[j] != host_frames[i + j]: 
                    synced = False
                    break
            
            # Adjust frames
            if synced == True:
                synced_frames = host_frames[i: ]
                break

        return synced_frames

    # TODO need to isolate noise, get rid of sinusoidal wave from power
    def signal_processing(self):
        """
        Assuming the length of a sinusoid period
        """
        pass

    def gen_key(self):
        pass

    def find_period_length(self, signal):
        """
        To be used on both devices.

        Client must use first to be able to share preamble period
        for frame synchronization. Host may use after shaving
        frames. 
        """
        length = start = None

        for i in signal:
            # Begin counting frames in period
            if start == None and length == None: # TODO improve naive approach
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
        # Partition up frames by sinusoidal period
        # This will give us our step on the array to go to a different period
        
        # Perform synchronization process using host preamble and devices frames
        self.sync()
        # Filter out sinusoidal wave, extracting only noise
        self.signal_processing()
        # Once processed, make further chunks to be used to create bits based on
        # commitment length
        self.gen_key()
        pass

    def device_protocol(self):
        pass

    def host_protocol(self):
        pass

    def host_protocol_threaded(self):
        # TODO host must send preamble for synchronization
        pass

