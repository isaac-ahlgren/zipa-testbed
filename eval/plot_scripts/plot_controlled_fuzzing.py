import matplotlib.pyplot as plt
import glob

from utils import parse_eval_directory

def plot_schurmann():
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data"
    SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_controlled_fuzz"
    contents = parse_eval_directory(SCHURMANN_DATA_DIRECTORY, SCHURMANN_CONTROLLED_FUZZING_STUB)

    # Regenerate 500 examples for SNR 40, 30, 20, 10, 5

    # Plot bit err vs parameter pick for window length

    # Plot bit err vs parameter pick for band length

    # Plot bar graph of correlation between bit err and window length and bit err and band length

def main():
    plot_schurmann()

if __name__ == "__main__":
    main()