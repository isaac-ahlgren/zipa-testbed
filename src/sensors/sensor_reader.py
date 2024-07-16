from multiprocessing import Process, Queue, Value
from typing import Any, List, Tuple

MAX_CLIENTS = 1024


class SensorReader:
    def __init__(self, sensor: Any) -> None:
        self.sensor = sensor
        self.queues: List[Tuple[Value, Queue]] = []
        self.sensor.start()
        self.poll_process = Process(target=self.poll, name=sensor.name)
        self.poll_process.start()

    def poll(self) -> None:
        while True:
            data = self.sensor.read()

            for flag, queue in self.queues:
                if flag.value == 1:
                    queue.put(data)

    def add_protocol_queue(self, status_queue: Tuple[Value, Queue]) -> None:
        # Terminate poll process to synchronize protocol queue
        self.poll_process.terminate()
        self.queues.append(status_queue)
        # Restart polling process
        self.poll_process = Process(target=self.poll, name=self.sensor.name)
        self.poll_process.start()

    def remove_protocol_queue(self, protocol_queue: Tuple[Value, Queue]) -> None:
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

    SAMPLE_RATE = 44_100
    CHUNK_SIZE = 1_024
    TIME_LENGTH = 3

    def sensor_thread(sensor):
        process = Process(target=sensor.poll)
        process.start()

    def getter_thread(sensor):
        print(f"[GETTER] In getter thread.")
        pipe = sensor_reader.create_pipe()
        print(f"[GETTER] Pipe created.")
        data = []
        for i in range(100):
            print(f"[GETTER] Sending message through pipe.")
            pipe.send(1)
            received_data = pipe.recv()  # Corrected to use recv() to receive data
            data.append(received_data)
            print(f"[GETTER] Received data.")
            print(data)
        print(f"[GETTER] Sending message through pipe.")
        pipe.send(-1)
        print(f"[GETTER] Data extracted:\n{data}\n")
        exit()

    test_sensor = TestSensor(
        SAMPLE_RATE, SAMPLE_RATE * TIME_LENGTH, CHUNK_SIZE, signal_type="random"
    )
    sensor_reader = SensorReader(test_sensor)
    protocol_pipe = sensor_reader.create_pipe()
    process = Process(target=getter_thread, args=[sensor_reader])
    process.start()
