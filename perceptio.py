import numpy as np
from sklearn.cluster import KMeans
import struct
import os

import multiprocessing as mp
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from corrector import Fuzzy_Commitment
from network import *
from reed_solomon import ReedSolomonObj

class Perceptio_Protocol:
    def __init__(
        self,
        sensor,
        key_length,
        parity_symbols,
        time_length,
        a,
        cluster_sizes_to_check,
        cluster_th,
        top_th,
        bottom_th,
        lump_th,
        conf_thresh,
        timeout,
        logger,
        verbose=True,
    ):
        self.sensor = sensor
        self.a = a
        self.cluster_sizes_to_check = cluster_sizes_to_check
        self.cluster_th = cluster_th
        self.top_th = top_th
        self.bottom_th = bottom_th
        self.lump_th = lump_th
        self.conf_threshold = conf_thresh

        self.name = "perceptio"
        self.timeout = timeout

        self.key_length = key_length
        self.parity_symbols = parity_symbols
        self.commitment_length = parity_symbols + key_length
        self.re = Fuzzy_Commitment(
            ReedSolomonObj(self.commitment_length, key_length), key_length
        )
        self.hash_func = hashes.SHA256()

        self.logger = logger

        self.time_length = time_length

        self.count = 0

        self.verbose = verbose


    def extract_context(self):
        signal = self.sensor.read(self.time_length)
        print(signal)
        fps = self.perceptio(signal, self.commitment_length, self.sensor.sensor.sample_rate,
                               self.a, self.cluster_sizes_to_check, self.cluster_th, self.bottom_th, self.top_th, self.lump_th)
        print(fps)
        return fps, signal

    def parameters(self, is_host):
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"a: {self.a}\n"
        parameters += f"cluster_sizes_to_check: {self.cluster_sizes_to_check}\n"
        parameters += f"cluster_th: {self.cluster_th}\n"
        parameters += f"top_th: {self.cluster_th}"
        parameters += f"bottom_th: {self.bottom_th}" 
        parameters += f"time_length: {self.time_length}\n"

    def device_protocol(self, host_socket):
        host_socket.setblocking(1)

        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("\nSending ACK")
        ack(host_socket)
        
        successes = 0
        iterations = 0
        current_key = bytes([0 for i in range(64)])
        while successes < self.conf_threshold:
            success = False

            if self.verbose:
                print("Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return


            if self.verbose:
                print("Extracting context\n")
            # Extract bits from sensor
            witnesses, signal = self.extract_context()

            if self.verbose:
                print("Commiting all the witnesses\n")
            # Create all commitments
            commitments = []
            keys = []
            hs = []
            for i in range(len(witnesses)):
                key, commitment = self.re.commit_witness(witnesses[i])
                commitments.append(commitment)
                keys.append(key)
                hs.append(self.hash_function(key))
            
            if self.verbose:
                print("Sending commitments\n")
            # Send all commitments
            send_commit(commitments, hs, host_socket)

            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)
        
            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return
            
            if self.verbose:
                print("Commitments recieved\n")
                print("Uncommiting with witnesses\n")
            key = self.find_commitment(commitments, hs, witnesses)

            # Commitment failed, try again
            if key == None:
                if self.verbose:
                    print("Witnesses failed to uncommit any commitment - alerting other device for a retry\n")
                success = False
                send_status(host_socket, success) # alert other device to status

                self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
                )
                iterations += 1

                continue

            if self.verbose:
                print("Witnesses succeeded in uncommiting a commitment - alerting other device to the success")

            # Check up on other devices status
            status = status_standby(host_socket, self.timeout)
            if status == None:
                if self.verbose:
                    print("No status recieved within time limit - early exit.\n\n")
                return
            elif status == False:
                success = False
                self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
                )
                iterations += 1
                continue

            # Key Confirmation Phase

            if self.verbose:
                print("Performing key confirmation\n")
 
            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(key + current_key)

            # Hash prederived key
            pd_key_hash = self.hash_function(key)

            # Send nonce message to host
            generated_nonce = self.send_nonce_msg_to_host(
                host_socket, pd_key_hash, derived_key
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                return

            # If hashes are equal, then it was successful
            if self.verify_mac_from_host(
                recieved_nonce_msg, generated_nonce, derived_key
            ):
                success = True
                successes += 1
                current_key = derived_key

            if self.verbose:
                print("Produced Key: " + str(derived_key))
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(iterations)
                )

            self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
            )

            iterations += 1

        if self.verbose:
            if successes / iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(iterations)
                    + " succeeded: "
                    + str(successes >= self.conf_threshold),
                )
            ]
        )
            
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
        device_ip_addr, device_port = device_socket.getpeername()

        # Exit early if no devices to pair with
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return
        if self.verbose:
            print("Successfully ACKed participating device")
            print()
        
        successes = 0
        iterations = 0
        current_key = bytes([0 for i in range(64)])
        while successes < self.conf_threshold:
            success = False

            if self.verbose:
                print("ACKing all participating devices")
            ack(device_socket)

            if self.verbose:
                print("Extracting context\n")
            # Extract bits from sensor
            witnesses, signal = self.extract_context()

            if self.verbose:
                print("Commiting all the witnesses\n")
            # Create all commitments
            commitments = []
            keys = []
            hs = []
            for i in range(len(witnesses)):
                key, commitment = self.re.commit_witness(witnesses[i])
                commitments.append(commitment)
                keys.append(key)
                hs.append(self.hash_func(key))
            
            if self.verbose:
                print("Sending commitments\n")
            # Send all commitments
            send_commit(commitments, hs, device_socket)

            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not commitment:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return
            
            if self.verbose:
                print("Commitments recieved\n")
                print("Uncommiting with witnesses\n")

            key = self.find_commitment(commitments, hs, witnesses)

            # Commitment failed, try again
            if key == None:
                if self.verbose:
                    print("Witnesses failed to uncommit any commitment - alerting other device for a retry\n")
                    
                success = False
                send_status(device_socket, success) # alert other device to status

                self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
                )
                iterations += 1

                continue

            if self.verbose:
                print("Witnesses succeeded in uncommiting a commitment - alerting other device to the success")

            # Check up on other devices status
            status = status_standby(device_socket, self.timeout)
            if status == None:
                if self.verbose:
                    print("No status recieved within time limit - early exit.\n\n")
                return
            elif status == False:
                success = False
                self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
                )
                iterations += 1
                continue

            # Key Confirmation Phase

            # Hash prederived key
            pd_key_hash = self.hash_function(key)

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                    print()
                return

            # Derive new key using previous key and new prederived key from fuzzy commitment
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(key + current_key)

            if self.verify_mac_from_device(
                recieved_nonce_msg, derived_key, pd_key_hash
            ):
                success = True
                successes += 1
                current_key = derived_key

            # Create and send key confirmation value
            self.send_nonce_msg_to_device(
                device_socket, recieved_nonce_msg, derived_key, pd_key_hash
            )

            self.logger.log(
                [
                    ("witness", "txt", witnesses),
                    ("commitment", "txt", commitment),
                    ("success", "txt", str(success)),
                    ("signal", "csv", signal),
                ],
                count=iterations,
                ip_addr=device_ip_addr,
            )

            if self.verbose:
                print(
                    "success: "
                    + str(success)
                    + ", Number of successes: "
                    + str(successes)
                    + ", Total number of iterations: "
                    + str(iterations)
                )
                print()

            iterations += 1

        if self.verbose:
            if successes / iterations >= self.auth_threshold:
                print(
                    "Total Key Pairing Success: auth - "
                    + str(successes / iterations)
                )
            else:
                print(
                    "Total Key Pairing Failure: auth - "
                    + str(successes / iterations)
                )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    "successes: "
                    + str(successes)
                    + " total_iterations: "
                    + str(iterations)
                    + " succeeded: "
                    + str(successes >= self.conf_threshold),
                )
            ]
        )

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()
    
    def ewma(self, signal, a):
        y = np.zeros(len(signal))

        y[0] = a*signal[0]
        for i in range(1,len(signal)):
            y[i] = a*signal[i] + (1-a)*y[i-1]
        return y

    def get_events(self, signal, a, bottom_th, top_th, lump_th):

        signal = self.ewma(np.abs(signal), a)

        print(signal)
        # Get events that are within the threshold
        events = []
        found_event = False
        beg_event_num = None
        for i in range(len(signal)):
            if not found_event and signal[i] >= bottom_th and signal[i] <= top_th:
                found_event = True
                beg_event_num = i
            elif found_event and (signal[i] < bottom_th or signal[i] > top_th):
                found_event = False
                found_event = None
                events.append((beg_event_num, i))
        if found_event:
            events.append((beg_event_num, i))

        i = 0
        while i < len(events)-1:
            if events[i+1][0] - events[i][1] <= lump_th:
                new_element = (events[i][0], events[i+1][1])
                events.pop(i)
                events.pop(i)
                events.insert(i, new_element)
            else:
                i += 1

        return events

    def get_event_features(self, events, signal):
        event_features = []
        for i in range(len(events)):
            length = events[i][1] - events[i][0]
            max_amplitude = np.max(signal[events[i][0]:events[i][1]])
            event_features.append((length, max_amplitude))
        return event_features

    def kmeans_w_elbow_method(self, event_features, cluster_sizes_to_check, cluster_th):
        if len(event_features) < cluster_sizes_to_check:
            # Handle the case where the number of samples is less than the desired number of clusters
            if self.verbose:
                print("Warning: Insufficient samples for clustering. Returning default label and k=1.")
            return np.zeros(len(event_features), dtype=int), 1

        km = KMeans(1, n_init='auto', random_state=0).fit(event_features)
        x1 = km.inertia_
        rel_inert = x1
    
        k = None
        labels = None
        inertias = [rel_inert]

        for i in range(1, cluster_sizes_to_check):
            labels = km.labels_

            km = KMeans(i, n_init='auto', random_state=0).fit(event_features)
            x2 = km.inertia_

            inertias.append(x2) 
            perc = (x1 - x2) / rel_inert

            x1 = x2
        
            # Break if reached elbow
            if perc <= cluster_th:
                k = i - 1
                break

            # Break if reached end
            if i == cluster_sizes_to_check - 1:
                labels = km.labels_
                k = i
                break

        return labels, k

    def group_events(self, events, labels, k):
        event_groups = [[] for i in range(k)]
        for i in range(len(events)):
            event_groups[labels[i]].append(events[i])
        return event_groups

    def gen_fingerprints(self, grouped_events, k, key_size, Fs):
        def floatToBits(f):
            s = struct.pack('>f', f)
            return struct.unpack('>l', s)[0]
        
        fp = []
        for i in range(k):
            event_list = grouped_events[i]
            key = ''
            for i in range(len(event_list)):
                interval = (event_list[i][1] - event_list[i][0])/Fs
                key += bin(floatToBits(interval))[2:]
        
            if len(key) >= key_size:
                key = key[:key_size]
                fp.append(key)
        return fp

    def perceptio(self, signal, key_size, Fs, a, cluster_sizes_to_check, cluster_th, bottom_th, top_th, lump_th):
        events = self.get_events(signal, a, bottom_th, top_th, lump_th)
        if len(events) < 2:
            if self.verbose:
                print("Warning: Insufficient samples for clustering. Skipping attempt.")
            return [], []
        event_features = self.get_event_features(events, signal)

        labels, k = self.kmeans_w_elbow_method(event_features, cluster_sizes_to_check, cluster_th)
 
        grouped_events = self.group_events(events, labels, k)
        fps = self.gen_fingerprints(grouped_events, k, key_size, Fs)

        return fps, grouped_events

    def find_commitment(self, commitments, hashes, fingerprints):
        key = None
        for i in range(len(fingerprints)):
            for j in range(len(commitments)):
                potential_key = self.re.decommit_witness(commitments[j], fingerprints[i])
                potential_key_hash = self.hash_func(potential_key)
                if constant_time.bytes_eq(potential_key_hash, hashes[j]):
                    key = potential_key
                    break
        return key
    
    def send_nonce_msg_to_device(
        self, connection, recieved_nonce_msg, derived_key, prederived_key_hash
    ):
        nonce = os.urandom(self.nonce_byte_size)

        # Concatenate nonces together
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len : pd_hash_len + self.nonce_byte_size
        ]
        concat_nonce = nonce + recieved_nonce

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(concat_nonce)
        tag = mac.finalize()

        # Construct nonce message
        nonce_msg = nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    def send_nonce_msg_to_host(self, connection, prederived_key_hash, derived_key):
        # Generate Nonce
        nonce = os.urandom(self.nonce_byte_size)

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(nonce)
        tag = mac.finalize()

        # Create key confirmation message
        nonce_msg = prederived_key_hash + nonce + tag

        send_nonce_msg(connection, nonce_msg)

        return nonce

    def verify_mac_from_host(self, recieved_nonce_msg, generated_nonce, derived_key):
        success = False

        recieved_nonce = recieved_nonce_msg[0 : self.nonce_byte_size]

        # Create tag of Nonce
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce + generated_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[self.nonce_byte_size :]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success

    def verify_mac_from_device(
        self, recieved_nonce_msg, derived_key, prederived_key_hash
    ):
        success = False

        # Retrieve nonce used by device
        pd_hash_len = len(prederived_key_hash)
        recieved_nonce = recieved_nonce_msg[
            pd_hash_len : pd_hash_len + self.nonce_byte_size
        ]

        # Generate new MAC tag for the nonce with respect to the derived key
        mac = hmac.HMAC(derived_key, self.hash_func)
        mac.update(recieved_nonce)
        generated_tag = mac.finalize()

        recieved_tag = recieved_nonce_msg[pd_hash_len + self.nonce_byte_size :]
        if constant_time.bytes_eq(generated_tag, recieved_tag):
            success = True
        return success


###TESTING CODE###
import socket
def device(prot):
    print("device")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))
    prot.device_protocol(s)

def host(prot):
    print("host")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(0)
    prot.host_protocol([conn])

if __name__ == "__main__":
    import multiprocessing as mp
    from test_sensor import Test_Sensor
    from sensor_reader import Sensor_Reader
    from nfs import NFSLogger
    prot = Perceptio_Protocol(Sensor_Reader(Test_Sensor(44100, 44100*400, 1024, signal_type='random')),
                                    8,
                                    4,
                                    44100*20,
                                    0.3,
                                    3,
                                    0.08,
                                    0.75,
                                    0.5,
                                    9500,
                                    5,
                                    10,
                                    NFSLogger(None, None, None, None, None, 1, "./data"),
    )
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()
