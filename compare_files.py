def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()

    max_len = max(len(file1_lines), len(file2_lines))
    differences = []

    for i in range(max_len):
        line1 = file1_lines[i] if i < len(file1_lines) else ''
        line2 = file2_lines[i] if i < len(file2_lines) else ''
        if line1 != line2:
            differences.append((i + 1, line1, line2))

    return differences

def print_differences(differences):
    if not differences:
        print("The files are identical.")
    else:
        print(f"The files have {len(differences)} differences:")
        for line_num, line1, line2 in differences:
            print(f"Line {line_num}:\nFile1: {line1.strip()}\nFile2: {line2.strip()}\n")

if __name__ == "__main__":
    # hardcoding it cause tbh does not matter
    file1 = './local_data/202405222148.csv'
    file2 = 'mnt/data/202405222148.csv'

    for 

    differences = compare_files(file1, file2)
    print_differences(differences)
