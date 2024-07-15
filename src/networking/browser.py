import ipaddress
import multiprocessing as mp
import socket
from multiprocessing.shared_memory import SharedMemory

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf


class ZIPA_Service_Browser:
    """
    Manages browsing of ZIPA (Zero Configuration IP Addressing) services over the network using Zeroconf.
    This class handles the discovery of ZIPA services and maintains a separate thread to run the service browsing.

    :param ip_addr: The local IP address of the device.
    :type ip_addr: str
    :param service_to_browse: The service type string to browse for, typically in the format '_service._proto.local.'.
    :type service_to_browse: str
    """

    def __init__(self, ip_addr, service_to_browse):
        """
        Initializes the service browser with the specified local IP address and service type to browse.
        """
        # Initialized to look for other devices with Avahi
        zeroconf = Zeroconf()
        # Find other devices that are participating in ZIPA
        self.listener = ZIPA_Service_Listener(ip_addr)
        self.browser = ServiceBrowser(zeroconf, service_to_browse, self.listener)
        # Run this on its own thread
        self.serv_browser_thread = mp.Process(target=self.browser.run)

    def start_thread(self):
        """
        Starts the service browsing thread.
        """
        self.serv_browser_thread.start()

    def get_ip_addrs_for_zipa(self):
        """
        Retrieves the list of IP addresses that are advertising ZIPA services.

        :returns: A list of IP addresses discovered.
        :rtype: list
        """
        # List that will hold discovered IP addresses advertising ZIPA
        ip_addrs = []
        # Prevent race conditions by thread locking the listener
        self.listener.mutex.acquire()
        # Retrieve the list of potential broadcasted ZIPA devices' IP address
        advertised_zipa_addrs = self.listener.advertised_zipa_addrs
        for i in range(len(advertised_zipa_addrs)):
            advertised_ip = advertised_zipa_addrs[i]
            # If the current IP address in the list is available, add to the list
            if advertised_ip != 0:
                ip_addrs.append(str(ipaddress.ip_address(advertised_ip)))
        # Release the lock
        self.listener.mutex.release()
        return ip_addrs


class ZIPA_Service_Listener(ServiceListener):
    """
    Listens for ZIPA services being advertised over the network and maintains a list of active services.
    It uses multiprocessing shared memory to track addresses across processes safely.

    :param ip_addr: The IP address of the device running this listener.
    :type ip_addr: str
    """
    def __init__(self, ip_addr):
        """
        Initializes the listener with the local device's IP address.
        """
        self.addr_list_len = 256
        # List to find devices able to perform ZIPA
        self.advertised_zipa_addrs = mp.shared_memory.ShareableList(
            [0 for i in range(self.addr_list_len)]
        )
        self.mutex = mp.Lock()
        # Get the devices IP address
        self.device_int_ip = int(ipaddress.ip_address(ip_addr))

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """
        Removes a service from the shared address list when it is no longer available.

        :param zc: The Zeroconf instance.
        :param type_: The service type.
        :param name: The service name.
        """
        # Prevents race conditions; locks process to one thread
        self.mutex.acquire()
        # Grab the target hostname
        host_name = name[: name.index(".")] + ".local"
        # Get its IP address
        dns_resolved_ip = socket.gethostbyname(host_name)
        int_ip = int(ipaddress.ip_address(dns_resolved_ip))

        for i in range(self.addr_list_len):
            # If the target IP address is found, remove it
            if int_ip == self.advertised_zipa_addrs[i]:
                self.advertised_zipa_addrs[i] = 0
                break
        self.mutex.release()

    # Needs this defined, but not necessary for our implementation
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """
        An empty implementation, required by the interface but not used in this application.
        """
        return

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """
        Adds a new ZIPA service to the shared address list when it is discovered.

        :param zc: The Zeroconf instance.
        :param type_: The service type.
        :param name: The service name.
        """
        self.mutex.acquire()
        host_name = name[: name.index(".")] + ".local"
        dns_resolved_ip = socket.gethostbyname(host_name)
        int_ip = int(ipaddress.ip_address(dns_resolved_ip))

        # If this isn't the current devices IP, it's gotta be another
        if int_ip != self.device_int_ip:
            # Iterate throught the list of available addresses
            for i in range(self.addr_list_len):
                # If an empty spot is found
                if self.advertised_zipa_addrs[i] == 0:
                    # Add the advertised address into the list
                    self.advertised_zipa_addrs[i] = int_ip
                    break
        self.mutex.release()
