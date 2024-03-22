import os
from datetime import datetime

import mysql.connector
import numpy as np


class NFSLogger:
    def __init__(self, user, password, host, database, nfs_server_dir, identifier):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier

    def log(self, data_tuples, count=None, ip_addr=None):
        # Ditching timestamp, else new files will get written all the time
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Using indexing instead
        for data in data_tuples:
            # TODO Ensure that file gets written to NFS mounted on pi
            filename = f"/mnt/data/{data[0]}_id{self.identifier}"  # File creation time will be recorded by NFS's linux box
            if count is not None:
                filename += f"_count{count}"
            if ip_addr is not None:
                filename += f"_ipaddr{ip_addr}"
            filename += f".{data[1]}"

        mode = "ab" if isinstance(data[2], bytes) else "a"

        if os.path.exists(filename):
            with open(filename, mode) as file:
                file.write(data[2])
        else:
            with open(filename, mode) as file:
                header = f"Timestamp: {timestamp}\n"

                if count is not None:
                    header += f"Count: {count}\n"

                if mode == "a":
                    file.write(header)
                else:
                    file.write(header.encode("utf-8"))

                # If the file is CSV and data is not a string, use numpy to save
                if data[1] == "csv" and not isinstance(data[2], str):
                    np.savetxt(
                        file, data[2], header=header.strip(), comments="", fmt="%s"
                    )
                elif mode == "wb":
                    file.write(data[2])  # Data should already be bytes
                else:
                    file.write(
                        str(data[2])
                    )  # Data is a string, no encoding needed in text mode
        # log file path to mysql database
        self._log_to_mysql([filename])

    def _log_to_mysql(self, file_paths):
        try:
            conn = mysql.connector.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.database,
            )
            cursor = conn.cursor()
            for file_path in file_paths:
                cursor.execute(
                    "INSERT INTO file_paths (file_path) VALUES (%s)", (file_path,)
                )
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
