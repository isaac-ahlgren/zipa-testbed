import random
import sys
import os
import copy

sys.path.insert(1, os.getcwd() + "/eval")

from eval_tools import calc_all_event_bits, load_controlled_signal
from signal_file import Signal_File, Event_File

def generate_events(seed, num_events, start_range, duration_range, max_duration):
    """
    Generate a list of sequential events.
    
    Parameters:
    - seed: The seed for the random number generator.
    - num_events: Number of events to generate.
    - start_range: Tuple (min, max) for the range of starting times.
    - duration_range: Tuple (min, max) for the range of event durations.
    
    Returns:
    - List of tuples representing events as (start_time, end_time).
    """
    rng = random.Random(seed)
    events = []
    current_time = rng.randint(*start_range)
    
    for _ in range(num_events):
        duration = rng.randint(*duration_range)
        end_time = current_time + duration

        if end_time > max_duration:
            break

        events.append((current_time, end_time))
        current_time = end_time + 1  # Ensure sequential non-overlapping events.
    
    return events

def func(events, event_sigs):
    print(events)
    return "bits"

def test_calc_all_event_bits():
    sf = Signal_File("./data/", "*.wav", load_func=load_controlled_signal, id="test")
    ref_signal1 = load_controlled_signal("./data/adversary_controlled_signal.wav")
    ref_signal2 = load_controlled_signal("./data/controlled_signal.wav")

    boundary1 = len(ref_signal1)
    boundary2 = len(ref_signal2)

    max_file_length = boundary1 + boundary2 - 2

    event_list1 = generate_events(0, 20, (0, max_file_length), (0, 10000), max_file_length)
    event_list2 = generate_events(0, 24, (0, max_file_length), (0, 10000), max_file_length)
    event_list3 = generate_events(0, 19, (0, max_file_length), (0, 10000), max_file_length)
    ef1 = Event_File(event_list1, copy.deepcopy(sf))
    ef2 = Event_File(event_list2, copy.deepcopy(sf))
    ef3 = Event_File(event_list3, copy.deepcopy(sf))

    calc_all_event_bits([ef1, ef2, ef3], func, 4)