import os
import sys
import time

import numpy as np

sys.path.insert(1, os.getcwd() + "/eval/")

from signal_file import Signal_File, Signal_Buffer, Wrap_Around_File, Noisy_File
from eval_tools import load_controlled_signal

def test_signal_file():
    sf = Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test")

    assert sf.get_id() == "test"

    ref_signal = load_controlled_signal("./data/controlled_signal.wav")

    received_samples = sf.read(100)
    ref_samples = ref_signal[:100]

    assert np.array_equal(received_samples, ref_samples)

    sf.reset()

    assert sf.start_sample == 0


    read_length = len(ref_signal) + 1
    received_samples = sf.read(read_length) # read more than in buffer

    assert read_length != len(received_samples) 
    assert sf.get_finished_reading() is True
 
    del sf
    sf = Signal_File(
        "./data/",
        "*.wav",
        load_func=load_controlled_signal,
        id="test")

    assert len(sf.files) == 2

    ref_signal = load_controlled_signal("./data/adversary_controlled_signal.wav")
    read_length = len(ref_signal) + 1
    received_samples = sf.read(read_length)

    assert read_length == len(received_samples)
    assert sf.get_finished_reading() is False
    assert sf.start_sample == 1
    assert sf.curr_file_name == "./data/controlled_signal.wav"

    del ref_signal
    del sf
    sf1 = Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test"
    )

    sf2 = Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test"
    )

    sf1.read(100)
    sf1.sync(sf2)
    assert sf1.start_sample == sf2.start_sample

def test_signal_buffer():
    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Signal_Buffer(signal.copy())

    samples = sb.read(100)
    ref_signal = signal[:100]
    assert np.array_equal(samples, ref_signal)
    assert sb.start_sample == 100

    sb.reset()
    assert sb.start_sample == 0

    read_length = len(signal) + 1
    received_samples = sb.read(read_length)

    assert read_length != len(received_samples) 
    assert sb.get_finished_reading() is True

def test_wrap_around():
    sf = Wrap_Around_File(Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test"),  wrap_around_limit=2)
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")

    read_length = len(ref_signal) + 100
    received_samples = sf.read(read_length)
    assert read_length == len(received_samples)
    assert sf.get_finished_reading() is False
    assert sf.sf.start_sample == 100

    received_samples = sf.read(2*read_length)
    assert read_length != len(received_samples)
    assert sf.get_finished_reading() is True

    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Wrap_Around_File(Signal_Buffer(signal), wrap_around_limit=2)
    read_length = len(ref_signal) + 100
    received_samples = sb.read(read_length)

    assert read_length == len(received_samples)
    assert sb.get_finished_reading() is False
    assert sb.sf.start_sample == 100

    received_samples = sb.read(2*read_length)
    assert read_length != len(received_samples)
    assert sb.get_finished_reading() is True

def test_noisy_signal():
    sf = Noisy_File(Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test"), 20)
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sf.read(read_length)
    assert len(received_samples) == read_length
    assert sf.get_finished_reading() is False
    assert sf.sf.start_sample == read_length
    assert not np.array_equal(ref_signal[:100], received_samples)

    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Noisy_File(Signal_Buffer(signal), 20)
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sb.read(read_length)
    assert len(received_samples) == read_length
    assert sb.get_finished_reading() is False
    assert sb.sf.start_sample == read_length
    assert not np.array_equal(ref_signal[:100], received_samples)
    
    sf = Noisy_File(Wrap_Around_File(Signal_File(
        "./data/",
        "controlled_signal.wav",
        load_func=load_controlled_signal,
        id="test"), 1), 20)
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sf.read(read_length)
    assert len(received_samples) == read_length
    assert sf.get_finished_reading() is False
    assert sf.sf.sf.start_sample == read_length
    assert not np.array_equal(ref_signal[:100], received_samples)