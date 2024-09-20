import matplotlib.pyplot as plt
import numpy as np
from utils import extract_from_contents, get_avg_ber_list, parse_eval_directory, bit_err_vs_parameter_plot, make_plot_dir

def bit_err_device_pairs(device_pairs, contents, param_label, file_name_stub, fig_dir, savefig=True):
    for pairs in device_pairs:
        legit1 = pairs[0] + "_bits"
        legit2 = pairs[1] + "_bits"
        adv = pairs[2] + "_bits"

        legit_pair_file_stub = f"{file_name_stub}_{pairs[0]}_{pairs[1]}_{param_label}"
        adv_pair_file_stub = f"{file_name_stub}_{pairs[0]}_{pairs[2]}_{param_label}"

        legit1_byte_list = extract_from_contents(contents, legit1)
        legit2_byte_list = extract_from_contents(contents, legit2)
        adv_byte_list = extract_from_contents(contents, adv)

        params = extract_from_contents(contents, "params")
        param_list = extract_from_contents(params, "window_length")

        legit_ber = get_avg_ber_list(legit1_byte_list, legit2_byte_list)
        adv_ber = get_avg_ber_list(legit1_byte_list, adv_byte_list)

        bit_err_vs_parameter_plot(legit_ber, param_list, f"Legitimate {pairs[0]} vs {pairs[1]} Using {param_label}", param_label, savefig=savefig, file_name=f"{legit_pair_file_stub}", fig_dir=fig_dir)
        bit_err_vs_parameter_plot(adv_ber, param_list, f"Adversary {pairs[0]} vs {pairs[2]} Using {param_label}", param_label, savefig=savefig, file_name=f"{adv_pair_file_stub}", fig_dir=fig_dir)

def plot_schurmann(savefigs=True):
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data/schurmann_real_fuzz"
    SCHURMANN_REAL_FUZZING_STUB = "schurmann_real_fuzz_day1"
    FIG_DIR_NAME_STUB = "schurmann_real_fuzz"
    
    DEVICE_PAIRS = [("10.0.0.238", "10.0.0.228", "10.0.0.239"),
                    ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
                    ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
                    ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
                    ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
                    ("10.0.0.234", "10.0.0.239", "10.0.0.237")]

    contents = parse_eval_directory(
        f"{SCHURMANN_DATA_DIRECTORY}",
        f"{SCHURMANN_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    bit_err_device_pairs(DEVICE_PAIRS, contents, "window_length", SCHURMANN_REAL_FUZZING_STUB, fig_dir, savefig=savefigs)
    bit_err_device_pairs(DEVICE_PAIRS, contents, "band_length", SCHURMANN_REAL_FUZZING_STUB, fig_dir, savefig=savefigs)


def main():
    plot_schurmann()


if __name__ == "__main__":
    main()
