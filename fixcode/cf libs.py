import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# Nama file input
input_filename = "data/GRUP 1_SAMPEL 2_D 0.2 us_skala 5_1.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)


# Fungsi untuk membaca data dari file CSV NIST
def read_nist_csv(filename):
    nist_data = pd.read_csv(filename)
    nist_wavelengths = nist_data["obs_wl_air(nm)"].values
    nist_intensities = nist_data["intens"].values
    return nist_wavelengths, nist_intensities


# Nama file NIST CSV untuk kalsium (ubah sesuai kebutuhan)
nist_filename = "expdata/CaI-CaII.csv"

# Baca data dari file CSV NIST untuk kalsium
nist_wavelengths, nist_intensities = read_nist_csv(nist_filename)


# Fungsi untuk menghapus sinyal latar belakang
def remove_background(intensities, window_length=51, polyorder=3):
    background = savgol_filter(intensities, window_length, polyorder)
    corrected_intensities = intensities - background
    return background, corrected_intensities


# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Deteksi puncak
height_threshold = 0.001 * np.max(corrected_intensities)
distance_between_peaks = 1
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)


# Fungsi untuk mengidentifikasi puncak berdasarkan data NIST
def identify_peaks(
    measured_wavelengths,
    measured_intensities,
    nist_wavelengths,
    nist_intensities,
    tolerance=1.0,
):
    identified_peaks = []
    for wl in measured_wavelengths:
        closest_nist_wl = nist_wavelengths[np.argmin(np.abs(nist_wavelengths - wl))]
        if np.abs(closest_nist_wl - wl) <= tolerance:
            identified_peaks.append((wl, closest_nist_wl))
    return identified_peaks


# Identifikasi puncak berdasarkan data NIST
identified_peaks = identify_peaks(
    wavelengths[peaks], corrected_intensities[peaks], nist_wavelengths, nist_intensities
)

# Plot spektrum dengan puncak yang diidentifikasi
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, corrected_intensities, label="Spektrum Koreksi")
plt.plot(
    wavelengths[peaks], corrected_intensities[peaks], "x", label="Puncak Terdeteksi"
)
for measured_wl, nist_wl in identified_peaks:
    plt.axvline(
        x=measured_wl, color="r", linestyle="--", label=f"Puncak {nist_wl:.2f} nm"
    )
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.grid(True)
plt.title("Identifikasi Puncak dengan data NIST")
plt.show()

# Print puncak yang diidentifikasi
print("Puncak yang Diidentifikasi berdasarkan data NIST:")
for measured_wl, nist_wl in identified_peaks:
    print(
        f"Panjang Gelombang Terukur: {measured_wl:.2f} nm, Panjang Gelombang NIST: {nist_wl:.2f} nm"
    )
