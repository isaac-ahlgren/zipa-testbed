import matplotlib.pyplot as plt
import numpy as np
from utils import extract_from_contents, get_avg_ber_list, parse_eval_directory


def bit_err_vs_parameter_plot(ber_list, param_list, parameter_name):
    ziped_list = list(zip(ber_list, param_list))
    ziped_list.sort(key=lambda x: x[1])

    ord_ber_list = [x[0] for x in ziped_list]
    ord_param_list = [x[1] for x in ziped_list]

    plt.plot(ord_param_list, ord_ber_list)
    plt.show()


def plot_schurmann():
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data/schurmann_controlled_fuzz"
    SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_controlled_fuzz"

    for snr in [20.0, 40]:
        contents = parse_eval_directory(
            f"{SCHURMANN_DATA_DIRECTORY}",
            f"{SCHURMANN_CONTROLLED_FUZZING_STUB}_snr{snr}",
        )

        byte_list1 = extract_from_contents(contents, "legit_signal1_bits")
        byte_list2 = extract_from_contents(contents, "legit_signal2_bits")
        adv_byte_list = extract_from_contents(contents, "adv_signal_bits")

        legit_ber = get_avg_ber_list(byte_list1, byte_list2)
        adv_ber = get_avg_ber_list(byte_list1, adv_byte_list)

        params = extract_from_contents(contents, "params")
        window_lengths = extract_from_contents(params, "window_length")
        band_lengths = extract_from_contents(params, "band_length")
        band_portions = np.divide(band_lengths, window_lengths)

        bit_err_vs_parameter_plot(legit_ber, window_lengths, "Window Length")
        bit_err_vs_parameter_plot(adv_ber, window_lengths, "Window Length")

        bit_err_vs_parameter_plot(legit_ber, band_portions, "Band Length")
        bit_err_vs_parameter_plot(adv_ber, band_portions, "Band Length")

        # Regenerate 500 examples for SNR 40, 30, 20, 10, 5

        # Plot bit err vs parameter pick for window length

        # Plot bit err vs parameter pick for band length

        # Plot bar graph of correlation between bit err and window length and bit err and band length


def main():
    plot_schurmann()


if __name__ == "__main__":
    main()
