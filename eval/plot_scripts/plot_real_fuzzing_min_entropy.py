from utils import (
    extract_from_contents,
    get_min_entropy_list,
    make_plot_dir,
    parameter_plot,
    parse_eval_directory,
)


def min_entropy_devices(
    devices, contents, param_label, file_name_stub, fig_dir, savefig=True, parse_string="bits", key_length=128,
):
    for device in devices:
        device_id = device + "_" + parse_string

        device_file_stub = f"{file_name_stub}_{device}_{param_label}_minentropy"

        device_byte_list = extract_from_contents(contents, device_id)

        min_entropy_list = get_min_entropy_list(device_byte_list, key_length, 8)

        params = extract_from_contents(contents, "params")
        param_list = extract_from_contents(params, param_label)

        parameter_plot(
            min_entropy_list,
            param_list,
            f"{device} Using {param_label}",
            param_label,
            savefig=savefig,
            file_name=f"{device_file_stub}",
            fig_dir=fig_dir,
            ylabel="Min Entropy",
        )


def plot_schurmann(savefigs=True):
    SCHURMANN_DATA_DIRECTORY = (
        "/home/isaac/ZIPA-TESTBED-DATA/schurmann_data/schurmann_real_fuzz"
    )
    SCHURMANN_REAL_FUZZING_STUB = "schurmann_real_fuzz_day1"
    FIG_DIR_NAME_STUB = "schurmann_real_fuzz_min_entropy"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory(
        f"{SCHURMANN_DATA_DIRECTORY}",
        f"{SCHURMANN_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    min_entropy_devices(
        DEVICES,
        contents,
        "window_length",
        SCHURMANN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "band_length",
        SCHURMANN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )


def plot_miettinen(savefigs=True):
    MIETTINEN_DATA_DIRECTORY = (
        "/home/isaac/ZIPA-TESTBED-DATA/miettinen_data/miettinen_real_fuzz"
    )
    MIETTINEN_REAL_FUZZING_STUB = "miettinen_real_fuzz_day1"
    FIG_DIR_NAME_STUB = "miettinen_real_fuzz_min_entropy"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory(
        f"{MIETTINEN_DATA_DIRECTORY}",
        f"{MIETTINEN_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    min_entropy_devices(
        DEVICES,
        contents,
        "f_samples",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "w_samples",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "rel_thr",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "abs_thr",
        MIETTINEN_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
    )

def plot_fastzip(savefigs=True):
    FASTZIP_DATA_DIRECTORY = (
        "../fastzip/fastzip_data/fastzip_real_fuzz"
    )
    FASTZIP_REAL_FUZZING_STUB = "fastzip_real_fuzz"
    FIG_DIR_NAME_STUB = "fastzip_real_fuzz_min_entropy"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory(
        f"{FASTZIP_DATA_DIRECTORY}",
        f"{FASTZIP_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    min_entropy_devices(
        DEVICES,
        contents,
        "window_size",
        FASTZIP_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "overlap_size",
        FASTZIP_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "n_bits",
        FASTZIP_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "power_thr",
        FASTZIP_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "snr_thr",
        FASTZIP_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )

def plot_perceptio(savefigs=True):
    PERCEPTIO_DATA_DIRECTORY = (
        "../perceptio/perceptio_data/perceptio_real_fuzz"
    )
    PERCEPTIO_REAL_FUZZING_STUB = "perceptio_real_fuzz"
    FIG_DIR_NAME_STUB = "perceptio_real_fuzz_min_entropy"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory(
        f"{PERCEPTIO_DATA_DIRECTORY}",
        f"{PERCEPTIO_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    min_entropy_devices(
        DEVICES,
        contents,
        "top_th",
        PERCEPTIO_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "bottom_th",
        PERCEPTIO_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "lump_th",
        PERCEPTIO_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )
    min_entropy_devices(
        DEVICES,
        contents,
        "a",
        PERCEPTIO_REAL_FUZZING_STUB,
        fig_dir,
        savefig=savefigs,
        parse_string="eventbits",
    )

def main():
    #plot_schurmann()
    #plot_miettinen()
    plot_fastzip()
    plot_perceptio()


if __name__ == "__main__":
    main()
