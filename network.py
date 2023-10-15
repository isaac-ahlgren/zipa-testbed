import socket
import pickle

class Ad_Hoc_Network:
    def __init__(self, ip, other_ip, is_host):
        self.ip = ip
        self.other_ip = other_ip
        self.is_host = is_host
        self.personal_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.personal_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.personal_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if is_host:
            self.personal_sock.bind((ip, 5005))
            self.personal_sock.listen()
            (client_sock, addr) = self.personal_sock.accept()
            self.client_sock = client_sock
            self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print("Connection Made!")
        else:
            self.personal_sock.connect((other_ip, 5005))
            print("Connected to host")

    def send_start(self):
        print()
        print("Sending Start")
        print()
        self.client_sock.sendto("start".encode(), (self.other_ip, 5005))

    def get_ack(self):
        print()
        print("Polling For ACK")
        print()
        while (1):
            message, address = self.client_sock.recvfrom(8)
            if message is not None:
                break

    def send_commitment(self, C):
        print()
        print("Sending Commitment")
        print()
        pickled_C = pickle.dumps(C)
        self.client_sock.sendto(pickled_C, (self.other_ip, 5005))

    def get_start(self):
        print()
        print("Polling For Start")
        print()
        while (1):
            message, address = self.personal_sock.recvfrom(8)
            if message is not None:
                break

    def send_ack(self):
        print()
        print("Sending Ack")
        print()
        self.personal_sock.sendto("ack".encode(), (self.other_ip, 5005))

    def get_commitment(self, bytes_needed):
        print()
        print("Waiting For Commitment")
        print()
        while (1):
            message, address = self.personal_sock.recvfrom(bytes_needed)
            if message is not None:
                break
        print()
        print("Commitment Recieved")
        print()
        C = pickle.loads(message)
        return C
