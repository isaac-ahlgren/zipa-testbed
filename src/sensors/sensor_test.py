from multiprocessing import Process, Queue, Value
from typing import Any, List, Tuple

class SensorReaderTest:
    def __init__(self, sensor: Any) -> None:
        """
        Initializes the SensorReader with a given sensor.

        :param sensor: The sensor object to be managed.
        """
        self.sensor = sensor
        self.queues: List[Tuple[Value, Queue]] = []
        self.sensor.start()
        self.poll_process = Process(target=self.poll, name=sensor.name)
        self.poll_process.start()

    def poll(self) -> None:
        """
        Constantly polls the sensor for data and sends this data to all active queues whose flags are set to active (1).
        """
        while True:
            data = self.sensor.read()

            for flag, queue in self.queues:
                if flag.value == 1:
                    queue.put(data)
        return data

    def add_protocol_queue(self, status_queue: Tuple[Value, Queue]) -> None:
        """
        Adds a new queue to the SensorReader for data distribution and restarts the polling process.

        :param status_queue: A tuple containing a multiprocessing Value as a flag and a multiprocessing Queue.
        """
        # Terminate poll process to synchronize protocol queue
        self.poll_process.terminate()
        self.queues.append(status_queue)
        # Restart polling process
        self.poll_process = Process(target=self.poll, name=self.sensor.name)
        self.poll_process.start()

    def remove_protocol_queue(self, protocol_queue: Tuple[Value, Queue]) -> None:
        """
        Removes a specified queue from the SensorReader and restarts the polling process.

        :param protocol_queue: The queue tuple to be removed.
        """
        # Terminate poll process to synchronize protocol queue
        self.poll_process.terminate()
        # Find and remove protocol queue
        index = self.queues.index(protocol_queue)
        self.queues.pop(index)
        # Restart polling process
        self.poll_process = Process(target=self.poll, name=self.sensor.name)
        self.poll_process.start()


if __name__ == "__main__":
    from test_sensor import TestSensor

    def sensor_thread(sensor):
        process = Process(target=sensor.poll)
        process.start()

    test_sensor = TestSensor(signal_type="random")
    sensor_reader = SensorReaderTest(test_sensor)

    print(sensor_reader)