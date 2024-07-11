import multiprocessing as mp
import time

from potential_solution import *


class Sender:
    def __init__(self, pipe):
        self.pipe = pipe
        self.queues = []
        self.process = mp.Process(target=self.send, name="Sender")
        self.mutex = mp.Lock()
        self.process.start()

    def send(self):
        while True:
            for entry in self.queues:
                flag, queue = entry

                if not flag.empty():
                    queue.put("Hello world")
                else:
                    print("No queues to put data into.")
                
            time.sleep(1)

    def add(self):
        status_queue = self.pipe.recv()
        self.process.terminate()
        # with self.mutex:
        self.queues.append(status_queue)
        self.process = mp.Process(target=self.send, name="Sender")
        self.process.start()

class Listener:
    def __init__(self, pipe):
        self.pipe = pipe
        self.queue = PicklableQueue()
        self.flag = PicklableQueue()

    def listen(self):
        target = time.time() + 10
        while True:
            print(f"[LISTENER] Recieved: {self.queue.get()}")

            if time.time() > target:
                self.flag.get()

    def send(self):
        self.pipe.send((self.flag, self.queue))


if __name__ == "__main__":
    sender_pipe, listener_pipe = mp.Pipe()

    print("Creating sender.")
    sender = Sender(sender_pipe)
    time.sleep(3)
    print("Creating listener.")
    listener = Listener(listener_pipe)

    print("Stopping sender to add shared queue.")
    listener.send()
    sender.add()
    print(f"Sender's queue list: {sender.queues}")
    print(f"Sender's flag queue empty? {sender.queues[0][0].empty()}")

    print("Listener readying for data")
    listener.flag.put(1)
    print(f"Sender's queue list: {sender.queues}")
    print(f"Sender's flag queue empty? {sender.queues[0][0].empty()}")

    print("Listener is beginning listening.")
    listener_process = mp.Process(target=listener.listen)
    listener_process.start()
