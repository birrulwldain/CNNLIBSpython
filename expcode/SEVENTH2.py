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


# Fungsi untuk menghitung luas kurva puncak dengan latar belakang sebagai batas
def calculate_peak_area_with_background(
    wavelengths, intensities, background, peak_index, window=5
):
    left_bound = max(peak_index - window, 0)
    right_bound = min(peak_index + window, len(wavelengths) - 1)
    peak_wavelengths = wavelengths[left_bound : right_bound + 1]
    peak_intensities = (
        intensities[left_bound : right_bound + 1]
        - background[left_bound : right_bound + 1]
    )
    area = np.trapz(peak_intensities, peak_wavelengths)
    return area


# Daftar panjang gelombang referensi untuk elemen yang biasanya ditemukan dalam cangkang telur
reference_lines = {
    "Ca": [393.37, 396.85, 422.67, 428.30],
    "Mg": [279.55, 280.27, 285.21],
    "P": [177.50, 214.91, 253.40],
    # Tambahkan elemen lain jika diperlukan
}

# Nama file input
input_filename = "../expdata/spectrum_data.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Deteksi puncak
height_threshold = 0.01 * np.max(corrected_intensities)
distance_between_peaks = 1
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Identifikasi puncak yang sesuai dengan spektrum referensi
identified_peaks = []
peak_areas = []
peak_labels = []

for element, lines in reference_lines.items():
    for line in lines:
        peak_index = np.argmin(np.abs(wavelengths[peaks] - line))
        if np.abs(wavelengths[peaks][peak_index] - line) < 1:  # Toleransi kesesuaian
            identified_peaks.append(peaks[peak_index])
            area = calculate_peak_area_with_background(
                wavelengths, intensities, background, peaks[peak_index]
            )
            peak_areas.append(area)
            peak_labels.append(element)

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
    label="Puncak Teridentifikasi",
)

# Tambahkan label atom pada puncak yang diidentifikasi
for i, peak in enumerate(identified_peaks):
    plt.text(
        wavelengths[peak],
        corrected_intensities[peak],
        peak_labels[i],
        verticalalignment="bottom",
    )

plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.grid(True)
plt.title("Deteksi Puncak yang Sesuai dengan Garis Spektral Referensi")
plt.savefig("identified_peaks_with_labels.pdf", format="pdf", dpi=300)
plt.show()

# Print identified peaks wavelengths, areas, and labels
print("Panjang Gelombang Puncak yang Diidentifikasi, Luas Kurvanya, dan Jenis Atom:")
for i, peak in enumerate(identified_peaks):
    print(
        f"Panjang Gelombang: {wavelengths[peak]:.5f} nm, Intensitas: {corrected_intensities[peak]:.5f}, Luas Kurva: {peak_areas[i]:.5f}, Atom: {peak_labels[i]}"
    )
