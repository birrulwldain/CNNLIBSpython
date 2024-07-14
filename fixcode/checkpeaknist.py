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


# Fungsi untuk membaca data dari file CSV NIST
def read_nist_csv(filename):
    nist_data = pd.read_csv(filename)

    def clean_string(s):
        return "".join([char for char in s if char.isdigit() or char == "."])

    nist_data["obs_wl_air(nm)"] = (
        nist_data["obs_wl_air(nm)"]
        .apply(lambda x: str(x).replace('="', "0").replace('"', "0"))
        .apply(clean_string)
        .astype(float)
    )
    nist_data["intens"] = (
        nist_data["intens"]
        .apply(lambda x: str(x).replace('="', "0").replace('"', "0"))
        .apply(clean_string)
        .astype(float)
    )
    # Terapkan fungsi pembersihan string ke kolom yang relevan

    nist_wavelengths = nist_data["obs_wl_air(nm)"].values
    nist_intensities = nist_data["intens"].values
    nist_element = nist_data["element"].values
    nist_num = nist_data["sp_num"].values  # Menyimpan informasi elemen atom
    return nist_wavelengths, nist_intensities, nist_element, nist_num


# Fungsi untuk menghapus sinyal latar belakang
def remove_background(intensities, window_length=51, polyorder=3):
    background = savgol_filter(intensities, window_length, polyorder)
    corrected_intensities = intensities - background
    return background, corrected_intensities


# Fungsi untuk mengidentifikasi puncak berdasarkan data NIST
def identify_peaks(
    measured_wavelengths,
    measured_intensities,
    nist_wavelengths,
    nist_intensities,
    nist_elements,
    nist_num,
    tolerance=0.1,
):
    identified_peaks = []
    for wl, inten in zip(measured_wavelengths, measured_intensities):
        closest_idx = np.argmin(np.abs(nist_wavelengths - wl))
        closest_nist_wl = nist_wavelengths[closest_idx]
        closest_nist_intensity = nist_intensities[closest_idx]
        closest_nist_element = nist_elements[closest_idx]
        closest_nist_num = nist_num[closest_idx]
        if np.abs(closest_nist_wl - wl) <= tolerance:
            identified_peaks.append(
                (
                    wl,
                    inten,
                    closest_nist_wl,
                    closest_nist_element,
                    closest_nist_num,
                    closest_nist_intensity,
                )
            )
    return identified_peaks


# Nama file input
input_filename = "data/GRUP 1_SAMPEL 3_D 0.2 us_skala 5_1.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Periksa data dari file ASC
print(
    f"Data dari file ASC: Wavelengths: {wavelengths[:5]}, Intensities: {intensities[:5]}"
)
0
# Nama file NIST CSV untuk kalsium (ubah sesuai kebutuhan)
nist_filename = "expdata/CaI-CaII.csv"

# Baca data dari file CSV NIST untuk kalsium
nist_wavelengths, nist_intensities, nist_element, nist_num = read_nist_csv(
    nist_filename
)

# Periksa data dari file CSV NIST
print(
    f"Data dari file NIST CSV: Wavelengths: {nist_wavelengths[:5]}, Intensities: {nist_intensities[:5]}"
)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Filter data ASC
valid_indices = intensities > 0
filtered_wavelengths = wavelengths[valid_indices]
filtered_intensities = intensities[valid_indices]

# Filter data NIST CSV
valid_nist_indices = nist_intensities > 0
filtered_nist_wavelengths = nist_wavelengths[valid_nist_indices]
filtered_nist_intensities = nist_intensities[valid_nist_indices]
filtered_nist_elements = nist_element[valid_nist_indices]
filtered_nist_num = nist_num[valid_nist_indices]

# Pastikan ada cukup data setelah melakukan filter
if len(filtered_wavelengths) < 2 or len(filtered_nist_wavelengths) < 2:
    raise ValueError("Tidak cukup data valid setelah filter.")

# Deteksi puncak
height_threshold = 0.01 * np.max(corrected_intensities)
distance_between_peaks = 1
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Identifikasi puncak berdasarkan data NIST
identified_peaks = identify_peaks(
    wavelengths[peaks],
    corrected_intensities[peaks],
    filtered_nist_wavelengths,
    filtered_nist_intensities,
    filtered_nist_elements,
    filtered_nist_num,
)

# Plot spektrum dengan puncak yang diidentifikasi
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, corrected_intensities, label="Spektrum Koreksi")
plt.plot(
    wavelengths[peaks], corrected_intensities[peaks], "x", label="Puncak Terdeteksi"
)
for (
    measured_wl,
    inten,
    nist_wl,
    nist_element,
    nist_num,
    nist_intensity,
) in identified_peaks:
    plt.axvline(
        x=measured_wl,
        color="r",
        linestyle="--",
        label=f"{nist_element} {nist_num} Puncak NIST {nist_wl:.5f} nm, Intensitas ASC: {inten:.2f}, Intensitas NIST: {nist_intensity:.2f}",
        alpha=0.5,
    )
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.grid(True)
plt.title("Identifikasi Puncak dengan Data NIST")
plt.show()

# Print puncak yang diidentifikasi
print("Puncak yang Diidentifikasi berdasarkan Data NIST:")
for (
    measured_wl,
    inten,
    nist_wl,
    nist_element,
    nist_num,
    nist_intensity,
) in identified_peaks:
    print(
        f"Panjang Gelombang Terukur: {measured_wl:.6f} nm, Intensitas ASC: {inten:.2f}, Panjang Gelombang NIST: {nist_wl:.6f} nm, Elemen: {nist_element} {nist_num}, Intensitas NIST: {nist_intensity:.2f}"
    )
