import matplotlib.pyplot as plt
import numpy as np


# Fungsi untuk membaca data dari file ASC
def read_from_asc(filename):
    wavelengths = []
    intensities = []
    with open(filename, "r") as f:
        # Lewati baris header
        next(f)
        for line in f:
            # Lewati baris kosong atau baris yang tidak memiliki dua elemen
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            wl, inten = parts
            wavelengths.append(float(wl))
            intensities.append(float(inten))
    return np.array(wavelengths), np.array(intensities)


# Fungsi untuk menormalisasi data terhadap intensitas maksimum
def normalize_max(intensities):
    return intensities / np.max(intensities)


# Fungsi untuk menyimpan data ke file ASC
def save_to_asc(filename, wavelengths, intensities):
    with open(filename, "w") as f:
        f.write("Wavelength (nm)\tIntensity (normalized)\n")
        for wl, inten in zip(wavelengths, intensities):
            f.write(f"{wl}\t{inten}\n")


# Nama file input dan output
input_filename = "../expdata/ba6.asc"
output_filename = "../expdata/normalized_spectrum.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Normalisasi intensitas
intensities_normalized = normalize_max(intensities)

# Simpan data yang telah dinormalisasi
save_to_asc(output_filename, wavelengths, intensities_normalized)

# Konfirmasi
print(f"Data spektral yang telah dinormalisasi disimpan dalam file {output_filename}")

# Visualisasi hasil normalisasi
plt.plot(wavelengths, intensities, label="Spektrum Asli")
plt.plot(wavelengths, intensities_normalized, label="Normalisasi Maksimum")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.title("Normalisasi terhadap Intensitas Maksimum")
plt.show()
