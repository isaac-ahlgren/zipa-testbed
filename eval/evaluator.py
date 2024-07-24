from eval_tools import Signal_Buffer, Signal_File


class Evaluator:
    def __init__(self, bit_gen_algo_wrapper):
        self.bit_gen_algo_wrapper = bit_gen_algo_wrapper
        self.legit_bits1 = []
        self.legit_bits2 = []
        self.adv_bits = []

    def evaluate(self, signals, trials):
        legit_signal1, legit_signal2, adv_signal = signals
        for i in range(trials):
            bits1 = self.bit_gen_algo_wrapper(legit_signal1)
            self.legit_bits1.append(bits1)

            bits2 = self.bit_gen_algo_wrapper(legit_signal2)
            self.legit_bits2.append(bits2)

            adv_bits = self.bit_gen_algo_wrapper(adv_signal)
            self.adv_bits.append(adv_bits)

            if isinstance(legit_signal1, Signal_Buffer) or isinstance(
                legit_signal1, Signal_File
            ):
                legit_signal1.sync(legit_signal2)

    def cmp_func(self, func, key_length):
        legit_bit_errs = []
        adv_bit_errs = []
        for i in range(len(self.legit_bits1)):
            legit_bit_err = func(self.legit_bits1[i], self.legit_bits2[i], key_length)
            adv_bit_err = func(self.legit_bits1[i], self.adv_bits[i], key_length)
            legit_bit_errs.append(legit_bit_err)
            adv_bit_errs.append(adv_bit_err)
        return legit_bit_errs, adv_bit_errs
