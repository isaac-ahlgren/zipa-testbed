
from schurmann import goldsig_plus_noise_eval_schurmann, controlled_signal_plus_noise_schurmann
from miettinen import goldsig_plus_noise_eval_miettinen, controlled_signal_plus_noise_miettinen
from perceptio import goldsig_plus_noise_eval_perceptio, controlled_signal_plus_noise_perceptio
from fastzip import goldsig_plus_noise_eval_fastzip, controlled_signal_plus_noise_fastzip
from iotcupid import goldsig_plus_noise_eval_iotcupid, controlled_signal_plus_noise_iotcupid

def test_snr_levels(func, snr_levels):
    legit_ber_list = []
    adv_ber_list = []
    for snr in snr_levels:
        legit_ber, adv_ber = func(target_snr=snr)
        legit_ber_list.append(legit_ber)
        adv_ber_list.append(adv_ber)
    return legit_ber_list, adv_ber_list