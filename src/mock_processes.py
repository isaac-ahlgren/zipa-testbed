import multiprocessing as mp
import queue
import time

from potential_solution import PicklableQueue


class Sender:
    def __init__(self):
        self.queues = []
        self.process = mp.Process(target=self.send, name="Sender")
        self.mutex = mp.Lock()
        self.process.start()

    def send(self):
        while True:
            for flag, queue in self.queues:
                if flag.value == 1:
                    queue.put("Hello world")
                else:
                    print("No queues to put data into.")

            time.sleep(1)

    def add(self, protocol_queue):
        status_queue = protocol_queue
        self.process.terminate()
        # with self.mutex:
        self.queues.append(status_queue)
        self.process = mp.Process(target=self.send, name="Sender")
        self.process.start()


class Listener:
    def __init__(self, identifier):
        self.queue = mp.Queue()
        self.flag = mp.Value("i", 0)
        self.identifier = str(identifier)

    def listen(self):
        target = time.time() + 10
        while True:
            try:
                print(f"[LISTENER {self.identifier}] Recieved: {self.queue.get()}")

                if time.time() > target:
                    self.flag.value = 0
            except queue.Empty:
                print("Queue is empty.")


if __name__ == "__main__":
    sender_pipe, listener_pipe = mp.Pipe()

    print("Creating sender.")
    sender = Sender()
    time.sleep(3)
    print("Creating listener.")
    listener_0 = Listener(0)
    listener_1 = Listener(1)

    print("Stopping sender to add shared queue.")
    # listener.send()
    sender.add((listener_0.flag, listener_0.queue))
    print(f"Sender's queue list: {sender.queues}")
    # print(f"Sender's flag queue empty? {sender.queues[0][0].empty()}")
    sender.add((listener_1.flag, listener_1.queue))
    print("Listener readying for data")
    listener_0.flag.value = 1

    print(f"Sender's queue list: {sender.queues}")
    # print(f"Sender's flag queue empty? {sender.queues[0][0].empty()}")

    print("Listener is beginning listening.")
    listener_process = mp.Process(target=listener_0.listen)
    listener_process.start()

    time.sleep(5)

    listener_1.flag.value = 1
    another_listener_process = mp.Process(target=listener_1.listen)
    another_listener_process.start()
