import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import extract_from_contents, get_avg_ber_list, parse_eval_directory, parameter_plot, make_plot_dir, find_best_parameter_choice


def create_plots(data_dir, controlled_fuzzing_stub, fig_dir_name_stub, plot_func, savefigs=True):
    if savefigs:
        fig_dir = make_plot_dir(fig_dir_name_stub)
    else:
        fig_dir = None

    for snr in [40, 30, 20, 10]:
        file_name_stub = f"{fig_dir_name_stub}_snr{snr}"
        contents = parse_eval_directory(
            f"{data_dir}",
            f"{controlled_fuzzing_stub}_snr{snr}",
        )

        byte_list1 = extract_from_contents(contents, "legit_signal1_bits")
        byte_list2 = extract_from_contents(contents, "legit_signal2_bits")
        adv_byte_list = extract_from_contents(contents, "adv_signal_bits")

        legit_ber = get_avg_ber_list(byte_list1, byte_list2)
        adv_ber = get_avg_ber_list(byte_list1, adv_byte_list)

        params = extract_from_contents(contents, "params")

        if savefigs:
            find_best_parameter_choice(legit_ber, adv_ber, params, file_name_stub, fig_dir)

        plot_func(legit_ber, adv_ber, params, file_name_stub, fig_dir, savefigs=savefigs)


def plot_schurmann(savefigs=True):
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data/schurmann_controlled_fuzz"
    SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_controlled_fuzz"
    FIG_DIR_NAME_STUB = "schurmann_controlled_fuzz"

    def plot_func(legit_ber, adv_ber, params, file_name_stub, fig_dir, savefigs=savefigs):
        window_lengths = extract_from_contents(params, "window_length")
        band_lengths = extract_from_contents(params, "band_length")
        band_portions = np.divide(band_lengths, window_lengths)

        parameter_plot(legit_ber, window_lengths, "Window Length Parameter Sweep Bit Error Plot for Legit", "Window Length",
                                  savefig=savefigs, file_name=f"{file_name_stub}_winlen_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, window_lengths, "Window Length Parameter Sweep Bit Error Plot for Adversary", "Window Length",
                                  savefig=savefigs, file_name=f"{file_name_stub}_winlen_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, band_portions, "Band Length Parameter Sweep Bit Error Plot for Legit", "Band Portion",
                                  savefig=savefigs, file_name=f"{file_name_stub}_bandlen_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, band_portions, "Band Length Parameter Sweep Bit Error Plot for Adversary", "Band Portion",
                                  savefig=savefigs, file_name=f"{file_name_stub}_bandlen_adv", fig_dir=fig_dir)
        
    create_plots(SCHURMANN_DATA_DIRECTORY, SCHURMANN_CONTROLLED_FUZZING_STUB, FIG_DIR_NAME_STUB, plot_func, savefigs=savefigs)

def plot_miettinen(savefigs=True):
    MIETTINEN_DATA_DIRECTORY = "../miettinen/miettinen_data/miettinen_controlled_fuzz"
    MIETTINEN_CONTROLLED_FUZZING_STUB = "miettinen_controlled_fuzz"
    FIG_DIR_NAME_STUB = "miettinen_controlled_fuzz"

    def plot_func(legit_ber, adv_ber, params, file_name_stub, fig_dir, savefigs=savefigs):
        f_samples = extract_from_contents(params, "f_samples")
        w_samples = extract_from_contents(params, "w_samples")
        rel_thr = extract_from_contents(params, "rel_thr")
        abs_thr = extract_from_contents(params, "abs_thr")

        parameter_plot(legit_ber, f_samples, "f Samples Parameter Sweep Bit Error Plot for Legit", "f Samples",
                                  savefig=savefigs, file_name=f"{file_name_stub}_fsamp_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, w_samples, "f Samples Parameter Sweep Bit Error Plot for Adversary", "f Samples",
                                  savefig=savefigs, file_name=f"{file_name_stub}_fsamp_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, f_samples, "w Samples Parameter Sweep Bit Error Plot for Legit", "w Samples",
                                  savefig=savefigs, file_name=f"{file_name_stub}_wsamp_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, w_samples, "w Samples Parameter Sweep Bit Error Plot for Adversary", "w Samples",
                                  savefig=savefigs, file_name=f"{file_name_stub}_wsamp_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, rel_thr, "Relative Threshold Parameter Sweep Bit Error Plot for Legit", "Relative Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_relthr_legit", fig_dir=fig_dir, range=(0,100))
        parameter_plot(adv_ber, rel_thr, "Relative Threshold Parameter Sweep Bit Error Plot for Adversary", "Relative Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_relthr_adv", fig_dir=fig_dir, range=(0,100))
        
        parameter_plot(legit_ber, abs_thr, "Absolute Threshold Parameter Sweep Bit Error Plot for Legit", "Absolute Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_absthr_legit", fig_dir=fig_dir, range=(0,100))
        parameter_plot(adv_ber, abs_thr, "Absolute Threshold Parameter Sweep Bit Error Plot for Adversary", "Absolute Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_absthr_adv", fig_dir=fig_dir, range=(0,100))

    create_plots(MIETTINEN_DATA_DIRECTORY, MIETTINEN_CONTROLLED_FUZZING_STUB, FIG_DIR_NAME_STUB, plot_func, savefigs=savefigs)

def plot_fastzip(savefigs=True):
    FASTZIP_DATA_DIRECTORY = "../fastzip/fastzip_data/fastzip_controlled_fuzz"
    FASTZIP_CONTROLLED_FUZZING_STUB = "fastzip_controlled_fuzz"
    FIG_DIR_NAME_STUB = "fastzip_controlled_fuzz"

    def plot_func(legit_ber, adv_ber, params, file_name_stub, fig_dir, savefigs=savefigs):
        window_size = extract_from_contents(params, "window_size")
        overlap_size = extract_from_contents(params, "overlap_size")
        n_bits = extract_from_contents(params, "n_bits")
        eqd_delta = extract_from_contents(params, "eqd_delta")
        ewma = extract_from_contents(params, "ewma")
        alpha = extract_from_contents(params, "alpha")
        remove_noise = extract_from_contents(params, "remove_noise")
        normalize = extract_from_contents(params, "normalize")
        power_th = extract_from_contents(params, "power_th")
        snr_th = extract_from_contents(params, "snr_th")

        parameter_plot(legit_ber, window_size, "Window Size Parameter Sweep Bit Error Plot for Legit", "Window Size",
                                  savefig=savefigs, file_name=f"{file_name_stub}_winsize_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, window_size, "Window Size Parameter Sweep Bit Error Plot for Adversary", "Window Size",
                                  savefig=savefigs, file_name=f"{file_name_stub}_winsize_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, overlap_size, "Overlap Size Parameter Sweep Bit Error Plot for Legit", "Overlap Size",
                                  savefig=savefigs, file_name=f"{file_name_stub}_overlapsize_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, overlap_size, "Overlap Size Parameter Sweep Bit Error Plot for Adversary", "Overlap Size",
                                  savefig=savefigs, file_name=f"{file_name_stub}_overlapsize_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, n_bits, "Number of Bits Parameter Sweep Bit Error Plot for Legit", "Number of Bits",
                                  savefig=savefigs, file_name=f"{file_name_stub}_nbits_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, n_bits, "Number of Bits Parameter Sweep Bit Error Plot for Adversary", "Number of Bits",
                                  savefig=savefigs, file_name=f"{file_name_stub}_nbits_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, eqd_delta, "Equidistance Delta Parameter Sweep Bit Error Plot for Legit", "Equidistance Delta",
                                  savefig=savefigs, file_name=f"{file_name_stub}_eqddelta_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, eqd_delta, "Equidistance Delta Parameter Sweep Bit Error Plot for Adversary", "Equidistance Delta",
                                  savefig=savefigs, file_name=f"{file_name_stub}_eqddelta_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, ewma, "Use EWMA Parameter Sweep Bit Error Plot for Legit", "Used EWMA",
                                  savefig=savefigs, file_name=f"{file_name_stub}_ewmause_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, ewma, "Use EWMA Parameter Sweep Bit Error Plot for Adversary", "Used EWMA",
                                  savefig=savefigs, file_name=f"{file_name_stub}_ewmause_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, alpha, "Alpha Parameter Sweep Bit Error Plot for Legit", "Alpha",
                                  savefig=savefigs, file_name=f"{file_name_stub}_alpha_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, alpha, "Alpha Threshold Parameter Sweep Bit Error Plot for Adversary", "Alpha",
                                  savefig=savefigs, file_name=f"{file_name_stub}_alpha_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, remove_noise, "Use Remove Noise Parameter Sweep Bit Error Plot for Legit", "Used Remove Noise",
                                  savefig=savefigs, file_name=f"{file_name_stub}_rmnoiseuse_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, remove_noise, "Use Remove Noise Parameter Sweep Bit Error Plot for Adversary", "Used Remove Noise",
                                  savefig=savefigs, file_name=f"{file_name_stub}_rmnoiseuse_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, normalize, "Use Normalize Parameter Sweep Bit Error Plot for Legit", "Used Normalize",
                                  savefig=savefigs, file_name=f"{file_name_stub}_normuse_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, normalize, "Use Normalize Parameter Sweep Bit Error Plot for Adversary", "Used Normalize",
                                  savefig=savefigs, file_name=f"{file_name_stub}_normuse_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, power_th, "Power Threshold Parameter Sweep Bit Error Plot for Legit", "Power Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_powth_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, power_th, "Power Threshold Parameter Sweep Bit Error Plot for Adversary", "Power Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_powth_adv", fig_dir=fig_dir)
        
        parameter_plot(legit_ber, snr_th, "SNR Threshold Parameter Sweep Bit Error Plot for Legit", "SNR Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_snrth_legit", fig_dir=fig_dir)
        parameter_plot(adv_ber, snr_th, "SNR Threshold Parameter Sweep Bit Error Plot for Adversary", "SNR Threshold",
                                  savefig=savefigs, file_name=f"{file_name_stub}_snrth_adv", fig_dir=fig_dir)

    create_plots(FASTZIP_DATA_DIRECTORY, FASTZIP_CONTROLLED_FUZZING_STUB, FIG_DIR_NAME_STUB, plot_func, savefigs=savefigs)

def main():
    plot_schurmann()
    plot_miettinen()
    plot_fastzip()


if __name__ == "__main__":
    main()
