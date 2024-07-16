import os


def count_files(directory):
    count = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            count += 1
    return count


# Contoh penggunaan
directory = "data/"
file_count = count_files(directory)
print(f"Jumlah file dalam folder: {file_count}")
