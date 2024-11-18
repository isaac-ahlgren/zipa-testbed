import os
import sys

import numpy as np

sys.path.insert(1, os.getcwd() + "/eval/")

from eval_tools import load_controlled_signal  # noqa: E402
from signal_file import (  # noqa: E402
    Event_File,
    Noisy_File,
    Signal_Buffer,
    Signal_File,
    Wrap_Around_File,
)


def test_signal_file():
    sf = Signal_File(
        "./data/", "controlled_signal.wav", load_func=load_controlled_signal, id="test"
    )

    assert sf.get_id() == "test"  # nosec

    ref_signal = load_controlled_signal("./data/controlled_signal.wav")

    received_samples = sf.read(100)
    ref_samples = ref_signal[:100]

    assert np.array_equal(received_samples, ref_samples)  # nosec

    sf.reset()

    assert sf.start_sample == 0  # nosec

    read_length = len(ref_signal) + 1
    received_samples = sf.read(read_length)  # read more than in buffer

    assert read_length != len(received_samples)  # nosec
    assert sf.get_finished_reading() is True  # nosec

    del sf
    sf = Signal_File("./data/", "*.wav", load_func=load_controlled_signal, id="test")

    assert len(sf.files) == 2  # nosec

    ref_signal = load_controlled_signal("./data/adversary_controlled_signal.wav")
    read_length = len(ref_signal) + 1
    received_samples = sf.read(read_length)

    assert read_length == len(received_samples)  # nosec
    assert sf.get_finished_reading() is False  # nosec
    assert sf.start_sample == 1  # nosec
    assert sf.curr_file_name == "./data/controlled_signal.wav"  # nosec

    del ref_signal
    del sf
    sf1 = Signal_File(
        "./data/", "controlled_signal.wav", load_func=load_controlled_signal, id="test"
    )

    sf2 = Signal_File(
        "./data/", "controlled_signal.wav", load_func=load_controlled_signal, id="test"
    )

    sf1.read(100)
    sf1.sync(sf2)
    assert sf1.start_sample == sf2.start_sample  # nosec


def test_signal_buffer():
    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Signal_Buffer(signal.copy())

    samples = sb.read(100)
    ref_signal = signal[:100]
    assert np.array_equal(samples, ref_signal)  # nosec
    assert sb.start_sample == 100  # nosec

    sb.reset()
    assert sb.start_sample == 0  # nosec

    read_length = len(signal) + 1
    received_samples = sb.read(read_length)

    assert read_length != len(received_samples)  # nosec
    assert sb.get_finished_reading() is True  # nosec


def test_wrap_around():
    sf = Wrap_Around_File(
        Signal_File(
            "./data/",
            "controlled_signal.wav",
            load_func=load_controlled_signal,
            id="test",
        ),
        wrap_around_limit=2,
    )
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")

    read_length = len(ref_signal) + 100
    received_samples = sf.read(read_length)
    assert read_length == len(received_samples)  # nosec
    assert sf.get_finished_reading() is False  # nosec
    assert sf.sf.start_sample == 100  # nosec

    received_samples = sf.read(2 * read_length)
    assert read_length != len(received_samples)  # nosec
    assert sf.get_finished_reading() is True  # nosec

    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Wrap_Around_File(Signal_Buffer(signal), wrap_around_limit=2)
    read_length = len(ref_signal) + 100
    received_samples = sb.read(read_length)

    assert read_length == len(received_samples)  # nosec
    assert sb.get_finished_reading() is False  # nosec
    assert sb.sf.start_sample == 100  # nosec

    received_samples = sb.read(2 * read_length)
    assert read_length != len(received_samples)  # nosec
    assert sb.get_finished_reading() is True  # nosec

    sf.reset()
    sf.set_global_index(len(ref_signal))
    test_signal = sf.read(40000)
    assert np.array_equal(test_signal, ref_signal[0:40000])  # nosec
    assert sf.num_of_resets == 1  # nosec

    sf.set_global_index(3 * len(ref_signal))
    assert sf.get_finished_reading() is True  # nosec

    sf.set_global_index(0)
    assert sf.get_finished_reading() is False  # nosec


def test_noisy_signal():
    sf = Noisy_File(
        Signal_File(
            "./data/",
            "controlled_signal.wav",
            load_func=load_controlled_signal,
            id="test",
        ),
        20,
    )
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sf.read(read_length)
    assert len(received_samples) == read_length  # nosec
    assert sf.get_finished_reading() is False  # nosec
    assert sf.sf.start_sample == read_length  # nosec
    assert not np.array_equal(ref_signal[:100], received_samples)  # nosec

    signal = load_controlled_signal("./data/controlled_signal.wav")
    sb = Noisy_File(Signal_Buffer(signal), 20)
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sb.read(read_length)
    assert len(received_samples) == read_length  # nosec
    assert sb.get_finished_reading() is False  # nosec
    assert sb.sf.start_sample == read_length  # nosec
    assert not np.array_equal(ref_signal[:100], received_samples)  # nosec

    sf = Noisy_File(
        Wrap_Around_File(
            Signal_File(
                "./data/",
                "controlled_signal.wav",
                load_func=load_controlled_signal,
                id="test",
            ),
            1,
        ),
        20,
    )
    read_length = 100000
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")
    received_samples = sf.read(read_length)
    assert len(received_samples) == read_length  # nosec
    assert sf.get_finished_reading() is False  # nosec
    assert sf.sf.sf.start_sample == read_length  # nosec
    assert not np.array_equal(ref_signal[:100], received_samples)  # nosec


def test_set_global_index():
    sf = Signal_File("./data/", "*.wav", load_func=load_controlled_signal, id="test")
    ref_signal1 = load_controlled_signal("./data/adversary_controlled_signal.wav")
    ref_signal2 = load_controlled_signal("./data/controlled_signal.wav")

    read_length = 10000

    boundary = len(ref_signal1)

    # Test if it switches file properly (tests generate_file_and_index by looking ahead)
    sf.set_global_index(boundary)  # set global index to just after the first file

    # Read and test output from signal file with reference signal from raw buffer
    test_sig = sf.read(read_length)
    ref_sig = ref_signal2[:read_length]

    assert np.array_equal(ref_sig, test_sig)  # nosec

    # Test if it can switch back to the first file (tests look_up_file_and_index)
    sf.set_global_index(0)
    test_sig = sf.read(read_length)
    ref_sig = ref_signal1[:read_length]

    assert np.array_equal(ref_sig, test_sig)  # nosec

    # Test if it switched file properly again (tests look_up_file_and_index)
    sf.set_global_index(boundary)

    test_sig = sf.read(read_length)
    ref_sig = ref_signal2[:read_length]

    assert np.array_equal(ref_sig, test_sig)  # nosec

    # Create new signal file for fresh lookup table
    new_sf = Signal_File(
        "./data/", "*.wav", load_func=load_controlled_signal, id="test"
    )

    new_sf.read(boundary)
    new_sf.set_global_index(0)
    test_sig = new_sf.read(read_length)
    ref_sig = ref_signal1[:read_length]

    assert np.array_equal(ref_sig, test_sig)  # nosec

    new_sf.set_global_index(0)
    position = boundary + len(ref_signal2)
    new_sf.set_global_index(position)

    assert new_sf.get_finished_reading() is True  # nosec


def test_event_file():
    sf = Signal_File("./data/", "*.wav", load_func=load_controlled_signal, id="test")
    ref_signal1 = load_controlled_signal("./data/adversary_controlled_signal.wav")
    ref_signal2 = load_controlled_signal("./data/controlled_signal.wav")

    boundary = len(ref_signal1)

    event_list = [
        [0, 2 * 48000],
        [41 * 48000, 41 * 48000 + 100],
        [boundary - 48000, boundary + 48000],
    ]
    ef = Event_File(event_list, sf)

    events_timestamp, events = ef.get_events(2)

    assert len(events) == 2  # nosec
    assert np.array_equal(events[0], ref_signal1[0 : 2 * 48000])  # nosec
    assert np.array_equal(
        events[1], ref_signal1[41 * 48000 : 41 * 48000 + 100]
    )  # nosec

    events_timestamp, event = ef.get_events(2)

    assert len(event) == 1  # nosec
    assert np.array_equal(
        event[0], np.concatenate((ref_signal1[boundary - 48000 :], ref_signal2[:48000]))
    )  # nosec
    assert ef.get_finished_reading() is True  # nosec

    sf1 = Signal_File(
        "./data/", "adv*.wav", load_func=load_controlled_signal, id="test"
    )

    sf2 = Signal_File(
        "./data/", "con*.wav", load_func=load_controlled_signal, id="test"
    )

    event_list1 = [[0, 2 * 48000], [41 * 48000, 41 * 48000 + 100]]
    event_list2 = [[300, 2 * 48000], [3 * 48000, 3 * 48000 + 300]]

    ef1 = Event_File(event_list1, sf1)
    ef2 = Event_File(event_list2, sf2)

    ef1.sync(ef2)

    ef1_curr_event = ef1.get_current_event()
    ef2_curr_event = ef2.get_current_event()
    assert ef1_curr_event[0] == 41 * 48000  # nosec
    assert ef2_curr_event[0] == 300  # nosec
    assert ef1.event_index == 1  # nosec
    assert ef2.event_index == 0  # nosec

    ef1.reset()

    ef2.sync(ef1)

    ef1_curr_event = ef1.get_current_event()
    ef2_curr_event = ef2.get_current_event()
    assert ef1_curr_event[0] == 41 * 48000  # nosec
    assert ef2_curr_event[0] == 300  # nosec
    assert ef1.event_index == 1  # nosec
    assert ef2.event_index == 0  # nosec

    sf = Wrap_Around_File(
        Signal_File(
            "./data/",
            "controlled_signal.wav",
            load_func=load_controlled_signal,
            id="test",
        ),
        wrap_around_limit=2,
    )
    ref_signal = load_controlled_signal("./data/controlled_signal.wav")

    event_list = [[0, 2 * 48000], [len(ref_signal), len(ref_signal) + 2 * 48000]]

    ef = Event_File(event_list, sf)

    event1, event_sig1 = ef.get_events(1)

    event2, event_sig2 = ef.get_events(1)

    assert event1 != event2  # nosec
    assert np.array_equal(event_sig1, event_sig2)  # nosec
    assert np.array_equal(event_sig1[0], ref_signal[: 2 * 48000])  # nosec
