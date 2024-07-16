import glob
import os

# Definisikan pola pencarian file
file_pattern = "data/GRUP *SAMPEL *D 0.2 us_skala 5_*.asc"

# Cari semua file yang sesuai dengan pola
file_list = glob.glob(file_pattern)

# Print daftar file yang ditemukan untuk verifikasi
print(f"Daftar file yang ditemukan: {file_list}")

# Periksa apakah ada file yang ditemukan
if not file_list:
    raise ValueError("Tidak ada file yang sesuai dengan pola pencarian")

# Kelompokkan file berdasarkan grup, sampel, dan skala
grouped_files = {}

# Proses file yang ditemukan
for file in file_list:
    # Ekstrak nama file tanpa path
    file_name = os.path.basename(file)

    # Ekstrak grup, sampel, dan skala dari nama file
    parts = file_name.split("_")
    group = parts[0]  # Misal, 'GRUP 1'
    sample = parts[1]  # Misal, 'SAMPEL 1'
    scale = parts[-2]  # Misal, 'skala 5'

    # Buat kunci untuk dictionary berdasarkan grup, sampel, dan skala
    key = (group, sample, scale)

    # Tambahkan file ke dalam grup yang sesuai
    if key not in grouped_files:
        grouped_files[key] = []
    grouped_files[key].append(file)

# Sortir grup berdasarkan GRUP, SAMPEL, dan skala
sorted_groups = sorted(grouped_files.keys(), key=lambda x: (x[0], x[1], x[2]))

# Nama file untuk menyimpan hasil
output_file = "grouped_files.asc"

# Buka file untuk ditulis
with open(output_file, "w") as f:
    total_files = 0  # Inisialisasi total file

    # Tulis hasil pengelompokan file ke dalam file ASC
    for group in sorted_groups:
        file_count = len(grouped_files[group])  # Hitung jumlah file dalam grup
        total_files += file_count  # Tambahkan ke total file
        f.write(
            f"File untuk GRUP {group[0]}, SAMPEL {group[1]}, {group[2]} ({file_count} file):\n"
        )
        for filename in grouped_files[group]:
            f.write(f"  - {filename}\n")
        f.write("\n")  # Tambahkan baris kosong untuk pemisah antar grup

    # Tulis total file keseluruhan
    f.write(f"Total file: {total_files}\n")

print(f"Hasil pengelompokan file telah disimpan dalam: {output_file}")
