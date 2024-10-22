import matplotlib.pyplot as plt
import numpy as np
from utils import (
    extract_from_contents,
    find_best_parameter_choice,
    get_avg_ber_list,
    make_plot_dir,
    parameter_plot,
    parse_eval_directory,
)


def bit_err_device_pairs(
    device_pairs, contents, param_label, file_name_stub, fig_dir, savefig=True
):
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
        param_list = extract_from_contents(params, param_label)

        legit_ber = get_avg_ber_list(legit1_byte_list, legit2_byte_list)
        adv_ber = get_avg_ber_list(legit1_byte_list, adv_byte_list)

        if savefig:
            file_name = f"{file_name_stub}_{pairs[0]}_{pairs[1]}_{pairs[2]}"
            find_best_parameter_choice(legit_ber, adv_ber, params, file_name, fig_dir)

        parameter_plot(
            legit_ber,
            param_list,
            f"Legitimate {pairs[0]} vs {pairs[1]} Using {param_label}",
            param_label,
            savefig=savefig,
            file_name=f"{legit_pair_file_stub}",
            fig_dir=fig_dir,
        )
        parameter_plot(
            adv_ber,
            param_list,
            f"Adversary {pairs[0]} vs {pairs[2]} Using {param_label}",
            param_label,
            savefig=savefig,
            file_name=f"{adv_pair_file_stub}",
            fig_dir=fig_dir,
        )


def plot_schurmann(savefigs=True):
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data/schurmann_real_fuzz"
    SCHURMANN_REAL_FUZZING_STUB = "schurmann_real_fuzz_day1"
    FIG_DIR_NAME_STUB = "schurmann_real_fuzz"

    DEVICE_PAIRS = [
        ("10.0.0.238", "10.0.0.228", "10.0.0.239"),
        ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
        ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
        ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
        ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
        ("10.0.0.234", "10.0.0.239", "10.0.0.237"),
    ]

    contents = parse_eval_directory(
        f"{SCHURMANN_DATA_DIRECTORY}",
        f"{SCHURMANN_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "window_length",
        SCHURMANN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "band_length",
        SCHURMANN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )


def plot_miettinen(savefigs=True):
    MIETTINEN_DATA_DIRECTORY = "../miettinen/miettinen_data/miettinen_real_fuzz"
    MIETTINEN_REAL_FUZZING_STUB = "miettinen_real_fuzz_day1"
    FIG_DIR_NAME_STUB = "miettinen_real_fuzz"

    DEVICE_PAIRS = [
        ("10.0.0.238", "10.0.0.228", "10.0.0.239"),
        ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
        ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
        ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
        ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
        ("10.0.0.234", "10.0.0.239", "10.0.0.237"),
    ]

    contents = parse_eval_directory(
        f"{MIETTINEN_DATA_DIRECTORY}",
        f"{MIETTINEN_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "f_samples",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "w_samples",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "rel_thr",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    bit_err_device_pairs(
        DEVICE_PAIRS,
        contents,
        "abs_thr",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )


def main():
    # plot_schurmann()
    plot_miettinen()


if __name__ == "__main__":
    main()
