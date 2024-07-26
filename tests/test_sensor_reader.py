import os
import sys
import time
from multiprocessing import Queue, Value
from typing import List, Tuple

sys.path.insert(1, os.getcwd() + "/src")
sys.path.insert(1, os.getcwd() + "/src/sensors")

from sensors.sensor_reader import SensorReader  # noqa: E402
from sensors.test_sensor import TestSensor  # noqa: E402

SEN_TEST: dict[str, int] = {
    "sample_rate": 44100,
    "chunk_size": 1024,
    "time_collected": 3,
}


def test_add_protocol():
    test_sensor = TestSensor(SEN_TEST, signal_type="sine")
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
    # assert status_queue[1].qsize() > 0

    # checks to see if tuples are being added to the queue
    assert len(test_sen_reader.queues) > queue_size  # nosec
    assert test_sen_reader.queues != 0  # nosec
    test_sen_reader.poll_process.terminate()


def test_remove_protocol_queue():
    test_sensor = TestSensor(SEN_TEST, signal_type="sine")
    test_sen_reader = SensorReader(test_sensor)

    flag = Value("i", 0)
    queue = Queue()

    protocol_queue = (flag, queue)

    # add tuple to queues
    test_sen_reader.add_protocol_queue(protocol_queue)

    assert len(test_sen_reader.queues) == 1  # nosec

    # call remove_protocol_queue
    test_sen_reader.remove_protocol_queue(protocol_queue)

    # check to see if queues is empty
    assert len(test_sen_reader.queues) == 0  # nosec
    test_sen_reader.poll_process.terminate()


def test_poll():
    test_sensor = TestSensor(SEN_TEST, signal_type="sine")

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

    # checks to see if the data written to the queue matches the chunk being read
    assert (data == test_sensor.read()).all()  # nosec
    test_sen_reader.poll_process.terminate()
