import sys
import os
sys.path.insert(1, os.getcwd() + "/../../src/")

from protocols.shurmann import Shurmann_Siggs_Protocol

MICROPHONE_SAMPLING_RATE = 48000
ANTIALIASING_FILTER = 18000

def shurmann_wrapper_func(arr, window_length, band_len, sampling_freq, antialias_freq):
    return Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(arr, antialias_freq, sampling_freq, window_len=window_length, bands=band_len)