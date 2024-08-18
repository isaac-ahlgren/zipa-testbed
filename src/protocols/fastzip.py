import multiprocessing as mp

from protocols.protocol_interface import ProtocolInterface


# WIP
class FastZIP_Protocol(ProtocolInterface):
    def __init__(self, parameters, sensor, logger):
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "FastZIP_Protocol"
        self.wip = True
        self.count = 0

    def extract_context(self):
        pass

    def process_context(self):
        pass

    def parameters(self, is_host):
        pass

    def device_protocol(self, host):
        pass

    def host_protocol(self, device_sockets):
        # Log parameters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(True))])

        if self.verbose:
            print("Iteration " + str(self.count))
            print()
        for device in device_sockets:
            p = mp.Process(target=self.host_protocol_single_threaded, args=[device])
            p.start()

    def host_protocol_single_threaded(self, device_socket):
        pass
