import matplotlib.pyplot as plt
import glob

from utils import parse_eval_directory

def plot_schurmann():
    SCHURMANN_DATA_DIRECTORY = "../schurmann/schurmann_data"
    SCHURMANN_CONTROLLED_FUZZING_STUB = "schurmann_controlled_fuzz"
    contents = parse_eval_directory(SCHURMANN_DATA_DIRECTORY, SCHURMANN_CONTROLLED_FUZZING_STUB)
    print(contents)

def main():
    plot_schurmann()

if __name__ == "__main__":
    main()