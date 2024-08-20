
import argparse
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to other scripts
sys.path.insert(1, os.getcwd() + "/../schurmann")  # Gives us path to other scripts
sys.path.insert(1, os.getcwd() + "/../miettinen")  # Gives us path to other scripts
sys.path.insert(1, os.getcwd() + "/../perceptio")  # Gives us path to other scripts
sys.path.insert(1, os.getcwd() + "/../iotcupid")  # Gives us path to other scripts
sys.path.insert(1, os.getcwd() + "/../fastzip")  # Gives us path to other scripts
from schurmann import goldsig_plus_noise_eval_schurmann, controlled_signal_plus_noise_schurmann
from miettinen import goldsig_plus_noise_eval_miettinen, controlled_signal_plus_noise_miettinen
from perceptio import goldsig_plus_noise_eval_perceptio, controlled_signal_plus_noise_perceptio
from fastzip import goldsig_plus_noise_eval_fastzip, controlled_signal_plus_noise_fastzip
from iotcupid import goldsig_plus_noise_eval_iotcupid, controlled_signal_plus_noise_iotcupid

def get_command_line_args():
    p = argparse.ArgumentParser()
    p.add_argument("-snr_levels", nargs="+", type=float, default=[1,5,10,20,40])
    args = p.parse_args()

    return args.snr_levels

def test_snr_levels(funcs, snr_levels):
    legit_ber_matrix = []
    adv_ber_matrix = []
    for func in funcs:
        legit_ber_list = []
        adv_ber_list = []
        for snr in snr_levels:
            legit_ber, adv_ber = func(target_snr=snr, trials=100)
            legit_ber_list.append(legit_ber)
            adv_ber_list.append(adv_ber)
        legit_ber_matrix.append(legit_ber_list)
        adv_ber_matrix.append(adv_ber_list)
    
    return legit_ber_matrix, adv_ber_matrix

def gen_gold_sig_data(snr_levels):
    goldsig_eval_funcs = [goldsig_plus_noise_eval_schurmann.main,
                          goldsig_plus_noise_eval_miettinen.main,
                          goldsig_plus_noise_eval_perceptio.main,
                          goldsig_plus_noise_eval_fastzip.main,
                          goldsig_plus_noise_eval_iotcupid.main]

    legit_ber_matrix, adv_ber_matrix = test_snr_levels(goldsig_eval_funcs, snr_levels)
    return legit_ber_matrix, adv_ber_matrix

def gen_controlled_sig_data(snr_levels):
    controlled_sig_eval_funcs = [controlled_signal_plus_noise_schurmann.main,
                                 controlled_signal_plus_noise_miettinen.main,
                                 controlled_signal_plus_noise_perceptio.main,
                                 controlled_signal_plus_noise_fastzip.main,
                                 controlled_signal_plus_noise_iotcupid.main]

    legit_ber_matrix, adv_ber_matrix = test_snr_levels(controlled_sig_eval_funcs, snr_levels)
    return legit_ber_matrix, adv_ber_matrix

def convert_to_data_frame(matrix_values, snr_levels):
    matrix_values[0].insert(0, "Schurmann")
    matrix_values[1].insert(0, "Miettinen")
    matrix_values[2].insert(0, "Perceptio")
    matrix_values[3].insert(0, "FastZip")
    matrix_values[4].insert(0, "IoTCupid")

    column_names = ["Protocol"]
    for snr in snr_levels:
        column_names.append(f"{snr} SNR")

    df = pd.DataFrame(matrix_values, columns=column_names)
    return df

def plot_matrix(matrix_values, title, saveplot=True, filename=None):
    matrix_values.plot(x='Protocol', kind='bar', stacked=False, title=title, xlabel="Protocol", ylabel="Bit Error Rate")
    if saveplot:
        plt.savefig(filename)
    else:
        plt.show()

def main(snr_levels):
    goldsig_legit_ber_matrix, goldsig_adv_ber_matrix = gen_gold_sig_data(snr_levels)

    controlled_sig_legit_ber_matrix, controlled_sig_adv_ber_matrix = gen_controlled_sig_data(snr_levels)

    goldsig_legit_mat = convert_to_data_frame(goldsig_legit_ber_matrix, snr_levels)
    controlled_legit_mat = convert_to_data_frame(controlled_sig_legit_ber_matrix, snr_levels)

    goldsig_adv_mat = convert_to_data_frame(goldsig_adv_ber_matrix, snr_levels)
    controlled_adv_mat = convert_to_data_frame(controlled_sig_adv_ber_matrix, snr_levels)

    goldsig_legit_mat.to_csv("./plot_data/gold_sig_legit_snr_plot_data.csv")
    controlled_legit_mat.to_csv("./plot_data/controlled_sig_legit_snr_plot_data.csv")

    goldsig_adv_mat.to_csv("./plot_data/gold_sig_adv_snr_plot_data.csv")
    controlled_adv_mat.to_csv("./plot_data/controlled_sig_adv_snr_plot_data.csv")

    plot_matrix(goldsig_legit_mat, "Protocol BER by SNR Level Using Golden Signal", saveplot=True, filename="./plot_data/gold_sig_legit_snr.png")
    plot_matrix(controlled_legit_mat, "Protocol BER by SNR Level Using Controlled Signal", saveplot=True, filename="./plot_data/controlled_sig_legit_snr.png")
    plot_matrix(goldsig_adv_mat, "Protocol BER by SNR Level Using Adversary Signal", saveplot=True, filename="./plot_data/gold_sig_adv_snr.png")
    plot_matrix(controlled_adv_mat, "Protocol BER by SNR Level Using Controlled Adversary Signal", saveplot=True, filename="./plot_data/controlled_sig_adv_snr.png")  

if __name__ == "__main__":
    snr_levels = get_command_line_args()
    main(snr_levels)
    
