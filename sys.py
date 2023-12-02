from bit_extractor import Bit_Extractor
from network import Network
from corrector import Reed_Solomon
from galois import *
import time
import sys
import numpy as np

pickle_folder = "./pickled/"

class TR_Bit_Extract_System():
    def __init__(self, is_host, ip, other_ip, sample_rate, vector_num, eig_num, bins, seconds, exp_name, n, k, audio_source_name):
        self.ip = ip
        self.other_ip = other_ip
        self.sample_rate = sample_rate
        self.vector_num = vector_num
        self.eig_num = eig_num
        self.bins = bins
        self.seconds = seconds
#        self.net = Network(ip, other_ip, is_host)
        self.be = Bit_Extractor(sample_rate, vector_num, eig_num, bins, seconds, audio_source_name)
        self.re = Reed_Solomon(n, k)
        self.exp_name = exp_name
        self.count = 0

    def compare_bits(self, bits1, bits2):
        tot = 0
        for i in range(len(bits1)):
            if bits1[i] == bits2[i]:
                tot += 1
        return tot / len(bits1)

    def compare_auth_tokens(self, expected_poly, recieved_poly):
        ret = True
        for i in range(len(expected_poly.coeffs)):
            if expected_poly.coeffs[i] != recieved_poly.coeffs[i]:
                ret = False
                break
        return ret

    def extract_context(self):
        print()
        print("Extracting Audio")
        key, conv = self.be.extract_key()
#        print("Generated key: " + str(key))
        print()
        return key, conv

    def bit_agreement_exp_dev(self):
        
        while (1):
            # Wait for start from host
            self.net.get_start()
            
            # Sending ack that they can start
            self.net.send_ack()

            # Extract key from mic
            key, conv = self.extract_context()

            # Save bits for later evaluation
            np.save(pickle_folder + self.exp_name + "_other_mykey_" + str(self.count) + "_pickled.npy", key)

            # Send bits to compare agreement rate
            self.net.send_bits(key)

            # Wait for Codeword
            C = self.net.get_codeword(8192)

            # Decode Codeword
            dec_C = self.re.decode_message(C, key)

            # Send Authentication Token
            self.net.send_auth_token(dec_C)

            self.count += 1

    def bit_agreement_exp_host(self):
        while (1):
            authenticated = None
            time_taken = None
            
            tic = time.perf_counter()
            # Send start to device
            self.net.send_start()
            
            # Get Ack to make sure it isn't lagging from the previous iteration
            self.net.get_ack()

            # Extract key from mic
            key, conv = self.extract_context()

            # Save bits for later evaluation
            np.save(pickle_folder + self.exp_name + "_host_mykey_" + str(self.count) + "_pickled.npy", key)

            #tic1 = time.perf_counter()
            # Recieve bits to compare agreement rate
            other_bits = self.net.get_bits(len(key))
            agreement = self.compare_bits(key, other_bits)
            print("Agreement Rate: " + str(agreement))
            #toc1 = time.perf_counter()
            #wasted_time = toc1 - tic1

            # Create Codeword
            auth_tok, C = self.re.encode_message(key)
  
            # Sending Codeword
            self.net.send_codeword(C)

            # Recieve Authentication Token
            other_auth_tok = self.net.get_auth_token(8192)

            if self.compare_auth_tokens(auth_tok, other_auth_tok):
                print("Successful Authentication")
                authenticated = True
            else:
                print("Failed Authentication")
                authenticated = False

            toc = time.perf_counter()
            time_taken = toc - tic # - wasted_time
            print(time_taken)
            print(convergence)
            hash_map = {
                    "authenticated": authenticated,
                    "agreement": agreement,
                    "convergence": conv,
                    "time": time_taken
            }
            np.save(pickle_folder + self.exp_name + "_info_" + str(self.count) + "_pickled.npy", hash_map)
            self.count += 1
