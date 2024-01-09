import socket
import pickle
import multiprocessing as mp
from multiprocessing import shared_memory
import ipaddress
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import types
import select

class Network:
    def __init__(self, ip, service_to_browse):
        self.ip = ip

        # Set up queue between listening thread and main thread
        self.queue = mp.Queue()

        # Set up listening socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((ip, 5005))
        self.sock.setblocking(0)
        self.sock.listen()

        # Setup service browser thread
        self.browser = ZIPA_Service_Browser(ip, service_to_browse)
        print("Starting browser thread")
        print()
        self.browser.start_thread()

        # Start listening thread
        print("Starting listening thread")
        print()
        self.listening_thread = self.start_listening_thread()

    def start_listening_thread(self):
        p = mp.Process(target=self.listening_thread)
        p.start()
        return p

    def listening_thread(self):
        inputs = [self.sock]
        outputs = []
        while (1):
            readable, writable, exceptional = select.select(inputs, outputs, inputs)
            for s in readable:
                if s is self.sock:
                    connection, client_address = self.sock.accept()
                    connection.setblocking(0)
                    inputs.append(connection)
                else:
                    data = s.recv(1024)
                    if data:
                        sender_ip = int(ipaddress.ip_address(s.getpeername()[0]))
                        self.queue.put((sender_ip, data))
                    else:
                        if s in outputs:
                            outputs.remove(s)
                        inputs.remove(s)
                        s.close()

            for s in exceptional:
                inputs.remove(s)
                if s in outputs:
                    outputs.remove(s)
                s.close()

    def get_zipa_ip_addrs(self):
        return self.browser.get_ip_addrs_for_zipa()

    def send_msg(self, msg, ip):
        self.sock.sendto(msg, ip)

    def get_msg(self):
        if self.queue.empty():
            return None
        else:
            out = self.queue.get()
            print("MSG: " + str(out))
            return out

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
        msg_len = len(message)
        pickled_C = message[:msg_len-64] # Its sending a 512 bit hash so the last 64 bytes are for that
        h = message[msg_len-64:]
        C = pickle.loads(pickled_C)
        return C,h

class ZIPA_Service_Browser():
    def __init__(self, ip_addr, service_to_browse):
        zeroconf = Zeroconf()
        self.listener = ZIPA_Service_Listener(ip_addr)
        self.browser = ServiceBrowser(zeroconf, service_to_browse, self.listener)
        self.serv_browser_thread = mp.Process(target=self.browser.run)

    def start_thread(self):
        self.serv_browser_thread.start()

    def get_ip_addrs_for_zipa(self):
        ip_addrs = []
        self.listener.mutex.acquire()
        advertised_zipa_addrs = self.listener.zipa_addrs
        for i in range(len(advertised_zipa_addrs)):
            advertised_ip = advertised_zipa_addrs[i]
            if advertised_ip != 0:
                ip_addrs.append(str(ipaddress.ip_address(advertised_ip)))
        self.listener.mutex.release()
        return ip_addrs

class ZIPA_Service_Listener(ServiceListener):
    def __init__(self, ip_addr):
        self.addr_list_len = 256
        self.advertised_zipa_addrs = mp.shared_memory.ShareableList([0 for i in range(self.addr_list_len)])
        self.mutex = mp.Lock()
        self.device_int_ip = int(ipaddress.ip_address(ip_addr))

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.mutex.acquire()
        host_name = name[:name.index('.')] + ".local"
        dns_resolved_ip = socket.gethostbyname(host_name)
        int_ip = int(ipaddress.ip_address(dns_resolved_ip))
        for i in range(self.addr_list_len):
            if int_ip == self.advertised_zipa_addrs[i]:
                self.advertised_zipa_addrs[i] = 0
        self.mutex.release()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        return

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.mutex.acquire()
        host_name = name[:name.index('.')] + ".local"
        print(host_name)
        dns_resolved_ip = socket.gethostbyname(host_name)
        int_ip = int(ipaddress.ip_address(dns_resolved_ip))

        if int_ip != self.device_int_ip:
            for i in range(self.addr_list_len):
                if self.advertised_zipa_addrs[i] == 0:
                    self.advertised_zipa_addrs[i] = int_ip
        self.mutex.release()
