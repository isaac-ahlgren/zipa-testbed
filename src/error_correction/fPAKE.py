from liPAKE import LiPake
from cryptography.hazmat.primitives.asymmetric import ed448
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import algorithms, modes
from cryptography.hazmat.primitives.asymmetric import x448
from cryptography import exceptions
import RSSCodes
import os
import numpy as np

from networking.network import send_fpake_first_msg, fpake_first_msg_standby

class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self):
        """
        Initiates the fPAKE protocol with a given Password;
        n Will be the length of the given Password
        :param weakPW: The Password as binary String like "010101" to iterate over it. The passwords need to be of same length
        :param n: security parameter n in bytes default is 1 which is 256 bit |0:128 bit securuty,  1 : 256 bit security
        :param connection: a Connection to send and receive data (needs to implement the @ConnectionInterface) or use the @IPConnection
        """
        self.VK = None
        self.hash = hashes.SHA256()
        self.symmAlgo = algorithms.AES
        self.mode = modes.CBC
        self.curve = x448
        self.ecpub = x448.X448PublicKey
        self.ed = ed448.Ed448PrivateKey
        self.edpub = ed448.Ed448PublicKey
        self.keySize = 32  # Keysize of the Curve448 public Key
        self.n = 32  # securityBits 256

    def host_protocol(self, pw, conn):

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key()
        vkBytes = vk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

        #Prepare for @LiPake exchange
        Ki = []
        key_array = bytearray()
        # Execute @LiPake over each bit of the password
        for i in pw:

            iv = os.urandom(16)  # We use fixed size of AES256 cause we can't get 128 Bit Keys with KDF
            lp = LiPake(pw=i, label=vkBytes, iv=iv, Hash=self.hash, mode=self.mode, curve=self.curve,
                        symmAlgo=self.symmAlgo, symmetricKeySize=self.keySize, n=self.n)
            Xs, l = lp.getX()

            # Send generated X receive Y
            send_fpake_first_msg(conn, (Xs, l, iv))
            Y, l = conn.receive()

            k = lp.getKey(Y, l, self.ecpub, False)
            Ki.append(k)
            key_array += k

        # We use robust shamir secret sharing with reed solomon error correcting codes.
        # each key from lipake is 32 bit and we have 32 keys -> 32 * 32 will be the size of C
        secretkey = os.urandom(self.n)
        rss = RSSCodes.robustShamir(len(pw), 1, size=self.n)
        secretkey, C = rss.shamir_share(secretkey)
        E = []
        # E = C-K
        for i in range(len(pw)):
            E.append(RSSCodes.XORBytes(C[i], Ki[i]))
        #Sign our E with the secret key
        sig = signingKey.sign(RSSCodes.list_to_byte(E))

        #Send E + Signature + verification key + selected prime number to reconstruct
        conn.send((E, sig, vkBytes, rss.get_prime()))
        response = conn.recv()

        while response != "accepted".encode():
            conn.send((E, sig, vkBytes, rss.get_prime()))
            response = self.connection.receive()

        # Finish the protocol to get both sides to end the protocol and close connections
        if self.connection.receive() == "finalize".encode():
            conn.send("finalize".encode())
        self.connection.close()
        return (secretkey)

    def receive_protocol(self, benchmark=None):
        """
        Protocol the receiving end is running to exchange a symmetric key if the password of each party is similar
        :return: the negotiated key if password was similar to a certain degree
        """
        time = None
        n_time = 0
        n_time_total = 0
        c_time = 0
        c_time_total = 0
        if benchmark is not None:  # For Benchmarking only
            benchmark["LiPAKE"] = {}
            time = timer()
        if not self.connection.wait_for_connection():
            raise CouldNotConnectException()
        labelList = []
        Ki = []
        key_array = b""
        # LiPake iteration for each bit in password
        for i in self.pw:
            if time is not None:  # For Benchmarking only
                n_time = 0
                c_time = 0
                time.start_time()
            # get Init Vectors as well as labels and X_s for LiPake
            Xs, l1, iv = self.connection.receive()

            if time is not None:  # For Benchmarking only
                n_time += time.stop_time()
                time.start_time()

            if (self.VK == None):
                try:
                    self.VK = self.edpub.from_public_bytes(l1)
                except:
                    print("VK was not correct ! Abort the protocol")
                    raise NotAVerificationKeyException
            labelList.append(l1)
            lp = LiPake(pw=i, label=b"", iv=iv, Hash=self.hash, mode=self.mode, curve=self.curve,
                        symmAlgo=self.symmAlgo, symmetricKeySize=self.keySize, n=self.n)
            # Generate receivers Y_s
            Ys, l2 = lp.getX()

            if time is not None:  # For Benchmarking only
                c_time += time.stop_time()
                time.start_time()
            # Send Y_s with its label
            self.connection.send((Ys, l2))
            if time is not None:  # For Benchmarking only
                n_time += time.stop_time()
                time.start_time()

            k = lp.getKey(Xs, l1, self.ecpub, True)
            Ki.append(k)
            key_array += k

            if time is not None: # For Benchmarking only
                c_time += time.stop_time()
                c_time_total += c_time
                n_time_total += n_time
                benchmark["LiPAKE"]["run{:02d}".format(Ki.__len__())] = {"bit": i,
                                                                         "crypto_time": c_time,
                                                                         "network_time": n_time}
                time.start_time()

        accepted = False
        while not accepted:
            try:
                E, sig, vk, prime = self.connection.receive()
                accepted = True
                self.connection.send("accepted")
            except:
                self.connection.send("Failed")
                print("Failed retrying")
                # E, sig, vk, prime = self.connection.receive()

        if time is not None: # For Benchmarking only
            n_time = time.stop_time()
            benchmark["rss_network_time"] = n_time
            n_time_total += n_time
            time.start_time()
        # Create RSS scheme with prime and password
        rss = RSS.RSSCodes.robustShamir(self.pw.__len__(), 1, size=self.n, PRIME=prime)
        # Reconstruct Verification key of bytes and verify the send E
        if (vk != self.VK.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)):
            raise exceptions.InvalidKey()
            # print("VK is worng")
        try:
            self.VK.verify(sig, RSS.RSSCodes.list_to_byte(E))
        except:  # Cancel if signature if wrong
            self.connection.send("finalize")
            if self.connection.receive() == "finalize":
                self.connection.close()
            return os.urandom(self.n)
        C = []
        # Calculate C by trying to revers the XOR. If enough Kis are correct we can reconstruct the shared secret with RSS below
        for i in range(self.pw.__len__()):
            C.append(RSS.RSSCodes.XORBytes(E[i], Ki[i]))
        try:
            # use RSS to reconstruct secret key if enough Kis were correct
            U = rss.shamir_robust_reconstruct(C)
        except:
            # If RSS was not successful the key is random
            U = (os.urandom(self.n))
        if time is not None:  # For Benchmarking only
            c_time = time.stop_time()
            benchmark["rss_calc_time"] = c_time
            c_time_total += c_time
            benchmark["total_calculation_time"] = c_time_total
            benchmark["total_network_time"] = n_time_total

        # Close the connection and tell the other party to close the connection
        self.connection.send("finalize")
        if self.connection.receive() == "finalize":
            self.connection.close()

        return U
    
class NotAVerificationKeyException(Exception):
    pass