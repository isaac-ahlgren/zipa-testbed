import os

import numpy as np
import RSSCodes
from cryptography import exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed448, x448
from cryptography.hazmat.primitives.ciphers import algorithms, modes
from liPAKE import LiPake

from networking.network import send_fpake_msg, fpake_msg_standby, send_status, status_standby, SUCC


class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self, timeout):
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
        self.timeout = timeout

    def host_protocol(self, pw, conn):

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key()
        vkBytes = vk.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # Prepare for @LiPake exchange
        Ki = []
        key_array = bytearray()
        # Execute @LiPake over each bit of the password
        for i in pw:

            iv = os.urandom(
                16
            )  # We use fixed size of AES256 cause we can't get 128 Bit Keys with KDF
            lp = LiPake(
                pw=i,
                label=vkBytes,
                iv=iv,
                Hash=self.hash,
                mode=self.mode,
                curve=self.curve,
                symmAlgo=self.symmAlgo,
                symmetricKeySize=self.keySize,
                n=self.n,
            )
            Xs, l = lp.getX()

            # Send generated X receive Y
            send_fpake_msg(conn, [Xs, l, iv])
            msg = fpake_msg_standby(conn, self.timeout)
            Y = msg[0]
            l = msg[1]

            k = lp.getKey(Y, l, self.ecpub, False)
            Ki.append(k)
            key_array += k

        # We use robust shamir secret sharing with reed solomon error correcting codes.
        # each key from lipake is 32 bit and we have 32 keys -> 32 * 32 will be the size of C
        secretkey = os.urandom(self.n)
        rss = RSSCodes.robustShamir(len(pw), 1, size=self.n)
        secretkey, C = rss.shamir_share(secretkey)
        E = []
        for i in range(len(pw)):
            E.append(RSSCodes.XORBytes(C[i], Ki[i]))
            
        # Sign our E with the secret key
        sig = signingKey.sign(RSSCodes.list_to_byte(E))

        # Send E + Signature + verification key + selected prime number to reconstruct
        send_fpake_msg(conn, [E, sig, vkBytes, rss.get_primes()])

        status = status_standby(conn, self.timeout)

        if status != SUCC:
            secretkey = None

        return secretkey

    def device_protocol(self, pw, conn):
        """
        Protocol the receiving end is running to exchange a symmetric key if the password of each party is similar
        :return: the negotiated key if password was similar to a certain degree
        """
        labelList = []
        Ki = []
        key_array = bytearray()
        # LiPake iteration for each bit in password
        for i in self.pw:
            # get Init Vectors as well as labels and X_s for LiPake
            msg = fpake_msg_standby(conn, self.timeout)
            Xs = msg[0]
            l1 = msg[1]
            iv = msg[2]

            if self.VK == None:
                try:
                    self.VK = self.edpub.from_public_bytes(l1)
                except:
                    print("VK was not correct ! Abort the protocol")
                    raise NotAVerificationKeyException
    
            labelList.append(l1)
            lp = LiPake(
                pw=i,
                label=b"",
                iv=iv,
                Hash=self.hash,
                mode=self.mode,
                curve=self.curve,
                symmAlgo=self.symmAlgo,
                symmetricKeySize=self.keySize,
                n=self.n,
            )
            # Generate receivers Y_s
            Ys, l2 = lp.getX()

            # Send Y_s with its label
            send_fpake_msg(conn, [Ys, l2])

            k = lp.getKey(Xs, l1, self.ecpub, True)
            Ki.append(k)
            key_array += k

        msg = fpake_msg_standby(conn, self.timeout)
        E = msg[0]
        sig = msg[1]
        vk = msg[2]
        prime = msg[3]

        # Create RSS scheme with prime and password
        rss = RSSCodes.robustShamir(len(pw), 1, size=self.n, PRIME=prime)

        # Reconstruct Verification key of bytes and verify the send E
        if vk != self.VK.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ):
            raise exceptions.InvalidKey()
        
        try:
            self.VK.verify(sig, RSSCodes.list_to_byte(E))
        except:  # Cancel if signature if wrong
            send_status(conn, False)
            return None

        C = []
        # Calculate C by trying to revers the XOR. If enough Kis are correct we can reconstruct the shared secret with RSS below
        for i in range(self.pw.__len__()):
            C.append(RSSCodes.XORBytes(E[i], Ki[i]))
        try:
            # use RSS to reconstruct secret key if enough Kis were correct
            U = rss.shamir_robust_reconstruct(C)
        except:
            # If RSS was not successful the key is random
            U = os.urandom(self.n)

        send_status(conn, True)

        return U


class NotAVerificationKeyException(Exception):
    pass
