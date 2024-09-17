import matplotlib.pyplot as plt
import numpy as np
from utils import extract_from_contents, get_avg_ber_list, parse_eval_directory



def bit_err_vs_parameter_plot(ber_list, param_list, plot_name, param_label):
    ziped_list = list(zip(ber_list, param_list))
    ziped_list.sort(key=lambda x: x[1])

    ord_ber_list = [x[0] for x in ziped_list]
    ord_param_list = [x[1] for x in ziped_list]

    plt.plot(ord_param_list, ord_ber_list)
    plt.title(plot_name)
    plt.xlabel(param_label)
    plt.show()

def bit_err_device_pairs(device_pairs, contents, param_label):
    for pairs in device_pairs:
        legit1 = pairs[0]
        legit2 = pairs[1]
        adv = pairs[2]

        legit1_byte_list = extract_from_contents(contents, legit1)
        legit2_byte_list = extract_from_contents(contents, legit2)
        adv_byte_list = extract_from_contents(contents, adv)

        params = extract_from_contents(contents, "params")
        param_list = extract_from_contents(params, "window_length")

        legit_ber = get_avg_ber_list(legit1_byte_list, legit2_byte_list)
        adv_ber = get_avg_ber_list(legit1_byte_list, adv_byte_list)

        bit_err_vs_parameter_plot(legit_ber, param_list, f"Legitimate {legit1} vs {legit2} Using {param_label}", param_label)
        bit_err_vs_parameter_plot(adv_ber, param_list, f"Adversary {legit1} vs {adv} Using {param_label}", param_label)

def plot_schurmann():
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data/schurmann_real_fuzz/"
    SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_real_fuzz_day1"
    
    DEVICE_PAIRS = [("10.0.0.238_bits","10.0.0.228_bits","10.0.0.239_bits"),
                    ("10.0.0.231_bits","10.0.0.232_bits","10.0.0.239_bits"),
                    ("10.0.0.233_bits","10.0.0.236_bits","10.0.0.239_bits"),
                    ("10.0.0.227_bits","10.0.0.229_bits","10.0.0.237_bits"),
                    ("10.0.0.235_bits","10.0.0.237_bits","10.0.0.239_bits"),
                    ("10.0.0.234_bits","10.0.0.239_bits","10.0.0.237_bits")]

    contents = parse_eval_directory(
        f"{SCHURMANN_DATA_DIRECTORY}",
        f"{SCHURMANN_CONTROLLED_FUZZING_STUB}",
    )

    bit_err_device_pairs(DEVICE_PAIRS, contents, "window_length")

        # Regenerate 500 examples for SNR 40, 30, 20, 10, 5

        # Plot bit err vs parameter pick for window length

        # Plot bit err vs parameter pick for band length

        # Plot bar graph of correlation between bit err and window length and bit err and band length


def main():
    plot_schurmann()


if __name__ == "__main__":
    main()
