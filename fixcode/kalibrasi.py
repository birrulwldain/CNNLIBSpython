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

    nist_wavelengths = nist_data["obs_wl_air(nm)"].values
    nist_intensities = nist_data["intens"].values
    nist_element = nist_data["element"].values
    nist_num = nist_data["sp_num"].values
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


# Fungsi untuk kalibrasi atau koreksi posisi puncak
def calibrate_peaks(identified_peaks):
    calibrated_peaks = []
    for (
        measured_wl,
        inten,
        nist_wl,
        nist_element,
        nist_num,
        nist_intensity,
    ) in identified_peaks:
        # Hitung perbedaan posisi
        shift = nist_wl - measured_wl
        # Koreksi posisi puncak dalam spektrum LIBS
        calibrated_wl = measured_wl + shift
        calibrated_peaks.append((calibrated_wl, inten))
    return calibrated_peaks


# Nama file input
input_filename = "data/GRUP 1_SAMPEL 3_D 0.2 us_skala 5_1.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Nama file NIST CSV untuk kalsium (ubah sesuai kebutuhan)
nist_filename = "expdata/CaI-CaII.csv"

# Baca data dari file CSV NIST untuk kalsium
nist_wavelengths, nist_intensities, nist_element, nist_num = read_nist_csv(
    nist_filename
)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

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
    nist_wavelengths,
    nist_intensities,
    nist_element,
    nist_num,
)

# Kalibrasi posisi puncak
calibrated_peaks = calibrate_peaks(identified_peaks)

# Buat file ASC baru dengan spektrum yang telah dikalibrasi
output_filename = "data/GRUP_1_SAMPEL_3_D_kalibrasi.asc"
with open(output_filename, "w") as f:
    f.write("Wavelength (nm)\tIntensity\n")
    for wl, intensity in zip(wavelengths, intensities):
        f.write(f"{wl:.6f}\t{intensity:.6f}\n")

# Tampilkan spektrum yang telah dikalibrasi
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, intensities, label="Spektrum Awal")
for calibrated_wl, intensity in calibrated_peaks:
    plt.axvline(
        x=calibrated_wl,
        color="r",
        linestyle="--",
        label=f"Puncak Kalibrasi: {calibrated_wl:.2f} nm",
    )
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.grid(True)
plt.title("Spektrum dengan Puncak yang Dikalibrasi")
plt.savefig("spektrum_kalibrasi.png")
plt.show()

print(f"File ASC baru telah dibuat: {output_filename}")
