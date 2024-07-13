import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter


# Fungsi untuk membaca data dari file ASC
def read_from_asc(filename):
    wavelengths = []
    intensities = []
    with open(filename, "r") as f:
        next(f)  # Lewati baris header
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            wl, inten = parts
            wavelengths.append(float(wl))
            intensities.append(float(inten))
    return np.array(wavelengths), np.array(intensities)


# Fungsi untuk menghapus sinyal latar belakang
def remove_background(intensities, window_length=51, polyorder=3):
    background = savgol_filter(intensities, window_length, polyorder)
    corrected_intensities = intensities - background
    return background, corrected_intensities


# Fungsi untuk menghitung luas kurva puncak
def calculate_peak_area(wavelengths, intensities, peak_index, window=5):
    left_bound = max(peak_index - window, 0)
    right_bound = min(peak_index + window, len(wavelengths) - 1)
    peak_wavelengths = wavelengths[left_bound : right_bound + 1]
    peak_intensities = intensities[left_bound : right_bound + 1]
    area = np.trapz(peak_intensities, peak_wavelengths)
    return area


# data panjang gelombang garis spektral kalsium (Ca II)
ca_wavelengths = [393.37, 396.85, 422.67, 428.30]

# Nama file input
input_filename = "../data/Cu plate_skala 5_D 1 us_1.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Deteksi puncak
height_threshold = 3 * np.max(corrected_intensities)
distance_between_peaks = 3
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Identifikasi puncak yang sesuai dengan spektrum Ca
identified_peaks = []
peak_areas = []
for ca_wl in ca_wavelengths:
    peak_index = np.argmin(np.abs(wavelengths[peaks] - ca_wl))
    if np.abs(wavelengths[peaks][peak_index] - ca_wl) < 1:  # Toleransi kesesuaian
        identified_peaks.append(peaks[peak_index])
        area = calculate_peak_area(
            wavelengths, corrected_intensities, peaks[peak_index]
        )
        peak_areas.append(area)

# Plot spektrum dengan puncak yang diidentifikasi
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, corrected_intensities, label="Spektrum Koreksi")
plt.plot(
    wavelengths[peaks], corrected_intensities[peaks], "x", label="Puncak Terdeteksi"
)
plt.plot(
    wavelengths[identified_peaks],
    corrected_intensities[identified_peaks],
    "o",
    label="Puncak Ca II",
)
for ca_wl in ca_wavelengths:
    plt.axvline(x=ca_wl, color="r", linestyle="--", label=f"Ca II {ca_wl} nm")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.grid(True)
plt.title("Deteksi Puncak yang Sesuai dengan Garis Spektral Ca II")
plt.savefig("identified_ca_peaks.pdf", format="pdf", dpi=300)
plt.show()

# Print identified peaks wavelengths and their areas
print("Panjang Gelombang Puncak yang Diidentifikasi untuk Ca II dan Luas Kurvanya:")
for i, peak in enumerate(identified_peaks):
    print(
        f"Panjang Gelombang: {wavelengths[peak]:.5f} nm, Intensitas: {corrected_intensities[peak]:.5f}, Luas Kurva: {peak_areas[i]:.5f}"
    )
sn_threshold = 3
height_threshold = sn_threshold * noise

# Identifikasi puncak yang melebihi height threshold
peaks, properties = find_peaks(intensity, height=height_threshold)

# Plot data spektral dan puncak yang teridentifikasi
plt.plot(wavelength, intensity, label="Spektrum LIBS")
plt.plot(wavelength[peaks], intensity[peaks], "x", label="Puncak")
plt.axhline(y=height_threshold, color="r", linestyle="--", label="Ambang S/N = 3")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.show()

# Tampilkan informasi puncak
for peak in peaks:
    print(
        f"Puncak ditemukan pada panjang gelombang {wavelength[peak]} nm dengan intensitas {intensity[peak]}"
    )
