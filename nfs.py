import numpy as np
import mysql.connector
from datetime import datetime

class NFSLogger:
    def __init__(self, user, password, host, database, nfs_server_dir, identifier):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier

    def log(self, data_tuples, file_extension="txt", count=None):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for file_name, data in data_tuples:
            full_file_name = f"{self.nfs_server_dir}/{file_name}_id{self.identifier}_{timestamp}"
            if count is not None:
                full_file_name += f"_count{count}"
            full_file_name += f".{file_extension}"

            # Determine mode based on whether data is bytes or str
            mode = "wb" if isinstance(data, bytes) else "w"
            with open(full_file_name, mode) as file:
                # Write header
                header = f"Timestamp: {timestamp}\n"
                if count is not None:
                    header += f"Count: {count}\n"
                if mode == "w":
                    file.write(header)  # No need to encode for text mode
                else:
                    file.write(header.encode('utf-8'))  # Encode header for binary mode

                # If the file is CSV and data is not a string, use numpy to save
                if file_extension == "csv" and not isinstance(data, str):
                    np.savetxt(file, data, header=header.strip(), comments='', fmt='%s')
                elif mode == "wb":
                    file.write(data)  # Data should already be bytes
                else:
                    file.write(data)  # Data is a string, no encoding needed in text mode
            #log file path to mysql database
            self._log_to_mysql([full_file_name])

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
                cursor.execute("INSERT INTO file_paths (file_path) VALUES (%s)", (file_path,))
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

