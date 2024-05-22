import shutil

if __name__ == "__main__":
    origin = './local_data/test.txt'
    server = '/mnt/data/'

    with open(origin, 'w') as file:
            # Write content to the file
            file.write("I'm being sent to the NSF server!")

    shutil.copy(origin, server)

