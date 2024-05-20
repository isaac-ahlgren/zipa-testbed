import os
import shutil
import subprocess

import multiprocessing.shared_memory as shm
import multiprocessing as mp
from datetime import datetime

# import mysql.connector
import numpy as np

class NFSLogger:
    def __init__(self, user, password, host, database, nfs_server_dir, local_dir, identifier, use_local_dir=False):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier
        self.local_dir = local_dir
        self.use_local_dir = use_local_dir

        self.mutex = mp.Semaphore()
        self.shl = shm.ShareableList(
            # pad 60 bytes into each index 
            [" " * 60 for i in range(96)]
            )
        self.counter = mp.Value('i', 0)

    def log_signal(self, name, signal):
        # records name into a list
        if self.use_local_dir:
            directory = self.local_dir
        else:
            directory = self.nfs_server_dir

        # create new files every minute (%S for testing purposes )
        timestamp = datetime.now().strftime("%Y%m%d%H%S")

        # comment for testing purposes
        # filename = name + "_id" + str(self.identifier) + "_date" + timestamp + ".csv"
        filename = "TESTING_" + timestamp

        filepath = directory + filename

        print(filepath)

        # oops
        data = "\n".join(str(num) for num in signal) + "\n"

        try: # Write to NFS (or maybe its local doesnt matter!!!)
            # will this still create a file if unable to NFS 
            file = open(filepath, "a")
            file.write(data)
            file.close()

        except: # If it fails the write, write it to the local
            print("error sending to nfs server")
            
            source = self.local_dir + "name_" + filename # /name_file.csv
            # writing to local directory
            file = open(source)
            file.write(data)
            file.close()

            # writing to shared list
            self.mutex.acquire()

            self.counter.value += 1
            j = self.counter.value

            self.shl[j] = source

            self.mutex.release()

        while self.counter.value >= 0:
            i = self.counter.value

            source = self.shl[i]
            dest = os.path.join(self.nfs_server_dir, self.shl[i])

            # check internet connection before sending files over
            if self.ping_server() == True:
                with open(source, "r") as original, open(dest, "a") as copy:
                    for line in original:
                        copy.write(line)
                self.counter.value -= 1
            else:
                break

            os.remove(source)

    def ping_server(self, ip_address = '192.168.1.172'):
        try:
            output = subprocess.check_output(['ping', '-c', '1', ip_address], stderr=subprocess.STDOUT, universal_newlines=True)
            if "1 received" in output:
                return True
            else:
                return False
        except subprocess.CalledProcessError:
            return False
        

    def log(self, data_tuples, count=None, ip_addr=None):

        # Using indexing instead
        if self.use_local_dir:
            directory = self.local_dir
        else:
            directory = self.nfs_server_dir

        for name, file_ext, data in data_tuples:

            # Create filename
            filename = self.create_filename(directory, name, count, ip_addr, file_ext)

            # Determine mode based on whether data is bytes or str (it can be a list of bytes too)
            if isinstance(data, bytes) or (isinstance(data, list) and isinstance(data, bytes)):
                mode = "wb" 
            else:
                mode = "w"

            # Open File and perform operation
            with open(filename, mode) as file:
                # If the file is CSV and data is not a string, use numpy to save
                if file_ext == "csv" and not isinstance(data, str):
                    np.savetxt(
                        file, data, comments="", fmt="%s"
                    )
                elif mode == "wb":
                    if isinstance(data, list) and isinstance(data, bytes):
                        for byte_string in data:
                            file.write(byte_string)
                    else:
                        file.write(data)  # Data should already be bytes
                else:
                    file.write(
                        str(data)
                    )  # Data is a string, no encoding needed in text mode

            # log file path to mysql database
            if not self.local_dir:
                self._log_to_mysql([filename])

    def create_filename(self, directory, name, count, ip_addr, file_ext):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{directory}/{name}_id{self.identifier}_{timestamp}"
        if count is not None:
            filename += f"_count{count}"
        if ip_addr is not None:
                filename += f"_toipaddr{ip_addr}"
        filename += f".{file_ext}"


    # def _log_to_mysql(self, file_paths):
    #     try:
    #         conn = mysql.connector.connect(
    #             user=self.user,
    #             password=self.password,
    #             host=self.host,
    #             database=self.database,
    #         )
    #         cursor = conn.cursor()
    #         for file_path in file_paths:
    #             cursor.execute(
    #                 "INSERT INTO file_paths (file_path) VALUES (%s)", (file_path,)
    #             )
    #         conn.commit()
    #     except mysql.connector.Error as err:
    #         print(f"Error connecting to MySQL: {err}")
    #     finally:
    #         if conn and conn.is_connected():
    #             cursor.close()
    #             conn.close()
