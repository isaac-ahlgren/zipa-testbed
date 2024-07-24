from multiprocessing import Queue, Value
from typing import Any, List, Tuple
import sys
import os
import time

sys.path.insert(1, os.getcwd() + "/src")
sys.path.insert(1, os.getcwd() + "/src/sensors")

from sensors.test_sensor import TestSensor
from sensors.sensor_reader import SensorReader


SEN_TEST: dict[str, int] = {
    "sample_rate": 44100,
    "chunk_size": 1024,
    "time_collected": 3
}

class TestSensorWrapper:
    def __init__(self, test_sensor):
        self.name = test_sensor.name
        self.test_sensor = test_sensor
        self.output_data  = []

    # grabs data written from the sensor reader
    def read(self):
        chunk = self.test_sensor.read()
        self.output_data.append(chunk)

        # print(chunk)
        return chunk
    
    def start(self):
        pass

def test_add_protocol():
    test_sensor = TestSensor(
      SEN_TEST, signal_type="sine"
    )
    test_sen_reader = SensorReader(test_sensor)

    # Make queue tuple here
    flag = Value("i", 0)
    queue = Queue()
    queue_size = len(test_sen_reader.queues)

    status_queue = (flag, queue)

    # read from queue here (it'll hopefully be being filled by the sensor reader)
    test_sen_reader.add_protocol_queue(status_queue)

    flag.value = 1
    time.sleep(5)
    flag.value = -1

    test_sen_reader.queues.append(status_queue)

    # checks to see if the status_queue is being written to 
    assert status_queue[1].qsize() > 0

    # checks to see if tuples are being added to the queue
    assert len(test_sen_reader.queues) > queue_size
    
def test_remove_protocol_queue():
    test_sensor = TestSensor(
      SEN_TEST, signal_type="sine"
    )
    test_sen_reader = SensorReader(test_sensor)

    flag = Value("i", 0)
    queue = Queue()
    queue_size = len(test_sen_reader.queues)

    protocol_queue = (flag, queue)

    # add tuple to queues 
    test_add_protocol()

    # call remove_protocol_queue 
    test_sen_reader.remove_protocol_queue(protocol_queue)

    # check to see if queues is empty
    assert test_sen_reader.queues.qsize() == 0

def test_poll():
    test_sensor = TestSensorWrapper(TestSensor(
      SEN_TEST, signal_type="sine"
    ))

    test_sen_reader = SensorReader(test_sensor)

    # Make queue tuple here
    queue: List[Tuple[Value, Queue]] = []
    flag = Value("i", 0)
    queue = Queue()

    status_queue = (flag, queue)

    # read from queue here (it'll hopefully be being filled by the sensor reader)
    test_sen_reader.add_protocol_queue(status_queue)

    flag.value = 1
    time.sleep(5)
    flag.value = -1

    data = status_queue[1].get_nowait()

    # checks to see if the status_queue is being written to 
    assert (data == test_sensor.read()).all()
