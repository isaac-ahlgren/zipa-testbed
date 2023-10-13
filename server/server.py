from server_network import Server_Networking
import multiprocessing as mp
import numpy as np
import pickle

# Length in bytes
ID_LEN = 4
SIGNAL_TYPE_LEN = 10
META_DATA_LEN = 12

# Indexes into byte array for items
ID_INDEX = 0
SIGNAL_TYPE_INDEX = ID_LEN
META_DATA_INDEX = SIGNAL_TYPE_INDEX + SIGNAL_TYPE_LEN
SIGNAL_INDEX = META_DATA_INDEX + META_DATA_LEN

class Server:
    def __init__(self, ip, port, data_folder):
        self.queue = mp.Queue()
        self.network = Server_Networking(ip, port, self.queue)
        self.net_proc = mp.Process(target=self.network.start_service())
        self.output_folder = data_folder
        
    def start_network(self):
        self.net_proc.start()

    def start_backend(self):
        while (1):
            if not self.queue.empty():
                item = self.queue.get()
                identifier = pickle.loads(item[ID_INDEX:ID_INDEX+ID_LEN])
                signal_type = pickle.loads(item[SIGNAL_TYPE_INDEX:SIGNAL_TYPE_INDEX+SIGNAL_TYPE_LEN])
                meta_data = item[META_DATA_INDEX:META_DATA_INDEX+META_DATA_LEN]
                
                SIGNAL_LEN = pickle.loads(meta_data[0:4])
                BIT_LEN = pickle.loads(meta_data[4:8])

                iteration = pickle.loads(meta_data[8:12])

                signal = pickle.loads(item[SIGNAL_INDEX:SIGNAL_INDEX+SIGNAL_LEN])
                BIT_INDEX = SIGNAL_INDEX+SIGNAL_LEN
                bits = pickle.loads(item[BIT_INDEX:BIT_INDEX+BIT_LEN])
                np.savetxt(self.output_folder + "/" + signal_type + "_id" + str(identifier) + "_it" + str(iteration) + ".csv")
                np.savetxt(self.output_folder + "/" + signal_type + "bits_id" + str(identifier) + "_it" + str(iteration) + ".csv")

    def start(self):
        self.start_network()
        self.start_backend()

                
