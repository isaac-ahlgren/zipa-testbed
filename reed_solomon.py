import random
import ctypes
import os
import numpy as np

# Currently built to assume block size will be 8 bits
class ReedSolomonObj():
    def __init__(self, n, k):
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")
        if n > 255:
            raise Exception("n has to be n <= 255 to be valid for a Galois Field of 256")
        self.n = n
        self.k = k
        self.t = n - k
        rscode = ctypes.cdll.LoadLibrary(os.getcwd() + "/rscode-1.3/libecc.so")

        # Initialize library
        init = rscode.initialize_ecc
        init.restype = None
        init.argtypes = None
        init()

        # Initialize Reed-Solomon Instance
        initialize_rs_instance = rscode.initialize_rs_instance
        initialize_rs_instance.restype = ctypes.c_voidp
        initialize_rs_instance.argtypes = [ctypes.c_int]
        self.rs_instance = initialize_rs_instance(self.t)

        # Initialize Instance freeing fucntion
        self.free_rs_instance = rscode.free_rs_instance
        self.free_rs_instance.restype = None
        self.free_rs_instance.argtypes = [ctypes.c_voidp]

        # Initialize encode_data function
        self.encode_data = rscode.encode_data
        self.encode_data.restype = None
        self.encode_data.argtypes = [np.ctypeslib.ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                     ctypes.c_int,
                                     np.ctypeslib.ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                     ctypes.c_voidp]
        
        # Initialize decode_data
        self.decode_data = rscode.decode_data
        self.decode_data.restype = None
        self.decode_data.argtypes = [np.ctypeslib.ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                     ctypes.c_int,
                                     ctypes.c_voidp]

        # Initialize check_syndrome
        self.check_syndrome  = rscode.check_syndrome
        self.check_syndrome.restype = ctypes.c_int
        self.check_syndrome.argtypes = [ctypes.c_voidp]

        # Initialize corect_error_erasures
        self.correct_errors_erasures = rscode.correct_errors_erasures
        self.correct_errors_erasures.restype = ctypes.c_int
        self.correct_errors_erasures.argtypes = [np.ctypeslib.ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                                 ctypes.c_int,
                                                 ctypes.c_int,
                                                 np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                                 ctypes.c_voidp]

    def encode(self, key: bytearray) -> bytearray:
        np_key = np.frombuffer(key, dtype=np.int8)
        C = np.zeros(self.t + len(key), dtype=np.int8)
        self.encode_data(np_key, len(key), C, self.rs_instance)
        return bytearray(C.tobytes())

    def decode(self, C: bytearray) -> bytearray:
        np_C = np.frombuffer(C, dtype=np.int8)
        erasures = np.zeros(8, dtype=np.intc)

        self.decode_data(np_C, len(np_C), self.rs_instance)
        if self.check_syndrome(self.rs_instance):
            self.correct_errors_erasures(np_C, len(np_C), 0, erasures, self.rs_instance)
        
        key = np_C[0:self.k]
        return bytearray(key.tobytes())