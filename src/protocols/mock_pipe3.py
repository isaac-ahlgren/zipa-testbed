import multiprocessing as mp
import time

def protocol_thread(ready, data_queue, pipe):
    # Waiting for a simple message from the sensor
    print("Protocol: Waiting for initial message via pipe")
    message = pipe.recv()
    print(f"Protocol: Received '{message}'")

    # Simulate some setup delay
    print("Protocol: Initializing")
    time.sleep(2)
    print("Protocol: Ready to receive data")
    ready.value = 1  # Indicate readiness

    # Wait a bit to see if sensor sends data
    time.sleep(2)

    # Check for data in the queue
    if not data_queue.empty():
        received_data = data_queue.get()
        print(f"Protocol: Received data: {received_data}")
    else:
        print("Protocol: No data received")

    print("Protocol: Finished")

def sensor_thread(ready, data_queue, pipe):
    # Send a simple message to the protocol via pipe
    print("Sensor: Sending initial message via pipe")
    pipe.send("Starting up")

    print("Sensor: Waiting for protocol to be ready")
    while True:
        if ready.value == 1:
            print("Sensor: Protocol is ready, sending data")
            data_queue.put("Hello from Sensor!")
            break
        time.sleep(1)  # Check readiness periodically

    print("Sensor: Finished")

if __name__ == "__main__":
    # Initialize shared memory objects outside of the processes
    ready = mp.Value('i', 0)  # Initially not ready
    data_queue = mp.Queue()
    # Create a pipe
    protocol_conn, sensor_conn = mp.Pipe()

    # Set up processes for the protocol and the sensor
    protocol_process = mp.Process(target=protocol_thread, args=(ready, data_queue, protocol_conn))
    sensor_process = mp.Process(target=sensor_thread, args=(ready, data_queue, sensor_conn))

    # Start both processes
    protocol_process.start()
    sensor_process.start()

    # Join both processes to ensure they complete before the main process exits
    protocol_process.join()
    sensor_process.join()

