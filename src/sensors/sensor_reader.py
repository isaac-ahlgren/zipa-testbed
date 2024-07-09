from multiprocessing import Process, Lock

import queue

MAX_CLIENTS = 1024


class SensorReader:
    def __init__(self, sensor, pipe):
        self.sensor = sensor
        self.pipe = pipe
        self.queues = []
        self.mutex = Lock()
        self.sensor.start()
        self.poll_process = Process(target=self.poll, name=sensor.name + " POLL")
        self.poll_process.start()

    def poll(self):
        while True:
            data = self.sensor.extract()

            with self.mutex:
                for protocol_queue in self.queues:
                    status, protocol_queue = protocol_queue

                    if status.value == 1:
                        try:
                            protocol_queue.put(data)
                        except queue.Full:
                            continue


    def add_protocol_queue(self, protocol_queue):
        if self.poll_process.is_alive():
            # Graceful exit to prevent data corruption on shared memory objects
            self.poll_process.join()

        # Ensure no other object is interactivng with the queues
        with self.mutex:
            self.queues.append(protocol_queue)

        self.poll_process.start()

    def remove_protocol_queue(self, protocol_queue):
        if self.poll_process.is_alive():
            self.poll_process.join()

        with self.mutex:
            index = self.queues.index(protocol_queue)
            self.queues.pop(index)

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
