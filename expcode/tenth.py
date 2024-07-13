import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# Fungsi untuk membaca file ASC
def read_asc_file(file_path):
    data = np.loadtxt(
        file_path, skiprows=1
    )  # Membaca file ASC, melewati baris pertama jika perlu
    wavelength = data[:, 0]
    intensity = data[:, 1]
    return wavelength, intensity


# Fungsi untuk menghitung kebisingan (N)
def calculate_noise(intensity, noise_region_indices):
    noise_region = intensity[noise_region_indices]
    noise = np.std(noise_region)
    return noise


# File ASC yang akan dibaca
file_path = "../data/Cu plate_skala 5_D 1 us_1.asc"

# Membaca data spektral dari file ASC
wavelength, intensity = read_asc_file(file_path)

# Indeks wilayah tanpa puncak untuk menghitung kebisingan
noise_region_indices = np.concatenate((np.arange(0, 200), np.arange(800, 1000)))

# Hitung kebisingan (N)
noise = calculate_noise(intensity, noise_region_indices)

# Tetapkan ambang S/N = 3
sn_threshold = 3
height_threshold = sn_threshold * noise

# Jarak minimum antar puncak
min_peak_distance = 10  # Sesuaikan dengan data spektral

# Identifikasi puncak yang melebihi height threshold dan memenuhi jarak minimum antar puncak
peaks, properties = find_peaks(
    intensity, height=height_threshold, distance=min_peak_distance
)

# Identifikasi puncak terendah
if len(peaks) > 0:
    lowest_peak_index = np.argmin(intensity[peaks])
    lowest_peak = peaks[lowest_peak_index]
else:
    print("Tidak ada puncak yang memenuhi syarat yang ditemukan.")
    lowest_peak = None

# Plot data spektral dan puncak yang teridentifikasi
plt.plot(wavelength, intensity, label="Spektrum LIBS")
plt.plot(wavelength[peaks], intensity[peaks], "x", label="Puncak")
plt.axhline(y=height_threshold, color="r", linestyle="--", label="Ambang S/N = 3")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.show()

# Plot untuk memperbesar puncak terendah
if lowest_peak is not None:
    # Rentang panjang gelombang di sekitar puncak terendah
    zoom_range = 20  # Sesuaikan rentang ini
    min_wavelength = max(wavelength[lowest_peak] - zoom_range, wavelength[0])
    max_wavelength = min(wavelength[lowest_peak] + zoom_range, wavelength[-1])

    zoom_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)

    plt.plot(wavelength[zoom_indices], intensity[zoom_indices], label="Spektrum LIBS")
    plt.plot(
        wavelength[lowest_peak], intensity[lowest_peak], "rx", label="Puncak Terendah"
    )
    plt.xlabel("Panjang Gelombang (nm)")
    plt.ylabel("Intensitas")
    plt.title(f"Puncak Terendah pada {wavelength[lowest_peak]:.2f} nm")
    plt.legend()
    plt.show()
else:
    print("Tidak ada puncak yang bisa diperbesar.")
