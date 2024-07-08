import multiprocessing as mp

MAX_CLIENTS = 1024


class SensorReader:
    def __init__(self, sensor, pipe):
        self.sensor = sensor
        self.pipe = pipe
        self.send = []
        self.mutex = mp.Lock()
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=sensor.name)
        self.poll_process.start()

    def poll(self):
        print(f"[POLLING] Process started.")
        while True:
            print(self.send)
            # Holds a list of pipes to protocols
            data = self.sensor.extract()
            print(f"[POLLING] data:\n{data}\n")
            with self.mutex:
                print(f"[POLLING] sending out data")
                for i, sensor_pipe in enumerate(self.pipes):
                    if sensor_pipe.poll():
                        self.send[i] = sensor_pipe.recv()
                    if self.send[i] == 1:
                        print(f"Found a pipe to send data to")
                        sensor_pipe.send(data)



if __name__ == "__main__":
    from test_sensor import TestSensor

    SAMPLE_RATE = 44_100
    CHUNK_SIZE = 1_024
    TIME_LENGTH = 3

    def sensor_thread(sensor):
        process = mp.Process(target=sensor.poll)
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
    process = mp.Process(target=getter_thread, args=[sensor_reader])
    process.start()
