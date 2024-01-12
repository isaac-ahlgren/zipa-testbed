import ipaddress
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

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
        advertised_zipa_addrs = self.listener.advertised_zipa_addrs
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
                break
        self.mutex.release()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        return

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.mutex.acquire()
        host_name = name[:name.index('.')] + ".local"
        dns_resolved_ip = socket.gethostbyname(host_name)
        int_ip = int(ipaddress.ip_address(dns_resolved_ip))

        if int_ip != self.device_int_ip:
            for i in range(self.addr_list_len):
                if self.advertised_zipa_addrs[i] == 0:
                    self.advertised_zipa_addrs[i] = int_ip
                    break
        self.mutex.release()