import os
import subprocess  # nosec
import sys

sys.path.insert(1, os.getcwd() + "/src/signal_processing")
original_dir = os.getcwd()


def testing_basic_schurmann_eval_run():
    os.chdir(original_dir)
    os.chdir("./eval/schurmann/")
    output = subprocess.run(
        ["python3", "goldsig_eval_schurmann.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "goldsig_plus_noise_eval_schurmann.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "controlled_signal_plus_noise_schurmann.py", "-t", "1"]
    )  # nosec
    output.check_returncode()
    os.chdir(original_dir)


def testing_basic_miettinen_eval_run():
    os.chdir(original_dir)
    os.chdir("./eval/miettinen")
    output = subprocess.run(
        ["python3", "goldsig_eval_miettinen.py", "-kl", "4", "-t", "1"]  # nosec
    )
    output.check_returncode()

    output = subprocess.run(
        [
            "python3",
            "goldsig_plus_noise_eval_miettinen.py",
            "-kl",
            "4",
            "-t",
            "1",
        ]  # nosec
    )
    output.check_returncode()

    output = subprocess.run(
        [
            "python3",
            "controlled_signal_plus_noise_miettinen.py",
            "-kl",
            "4",
            "-t",
            "1",
        ]  # nosec
    )
    output.check_returncode()
    os.chdir(original_dir)


def testing_basic_perceptio_eval_run():
    os.chdir(original_dir)
    os.chdir("./eval/perceptio")
    output = subprocess.run(
        ["python3", "goldsig_eval_perceptio.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "goldsig_plus_noise_eval_perceptio.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "controlled_signal_plus_noise_perceptio.py", "-t", "1"]
    )  # nosec
    output.check_returncode()
    os.chdir(original_dir)


def testing_basic_iotcupid_eval_run():
    os.chdir(original_dir)
    os.chdir("./eval/iotcupid")
    output = subprocess.run(["python3", "goldsig_eval_iotcupid.py", "-t", "1"])  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "goldsig_plus_noise_eval_iotcupid.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "controlled_signal_plus_noise_iotcupid.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    os.chdir(original_dir)


def testing_basic_fastzip_eval_run():
    os.chdir(original_dir)
    os.chdir("./eval/fastzip")
    output = subprocess.run(["python3", "goldsig_eval_fastzip.py", "-t", "1"])  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "goldsig_plus_noise_eval_fastzip.py", "-t", "1"]
    )  # nosec
    output.check_returncode()

    output = subprocess.run(
        ["python3", "controlled_signal_plus_noise_fastzip.py", "-t", "1"]
    )  # nosec
    output.check_returncode()
    os.chdir(original_dir)
