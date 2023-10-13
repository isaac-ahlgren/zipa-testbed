import socket
import selectors
import types
import multiprocessing as mp

class Server_Networking:
    def __init__(self, ip, port, queue):
        self.sel = selectors.DefaultSelector()
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.bind((ip, port))
        self.lsock.listen()
        self.lsock.setblocking(False)
        self.sel.register(self.lsock, selectors.EVENT_READ, data=None)
        self.queue = queue

    def start_service(self):
        try:
            while (1):
                events = self.sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        self.accept_wrapper(key.fileobj)
                    else:
                        self.service_connection(key, mask)
        finally:
            sel.close()

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()
        print("Accepted connection from " + str(addr))
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selector.EVENT_WRITE
        self.sel.register(conn, events, data=data)

    def service_connection(key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)
            if recv_data:
                self.queue.put(data)
            else:
                print("Closing connection to " + str(data.addr))
                sel.unregsiter(sock)
                sock.close()
            
