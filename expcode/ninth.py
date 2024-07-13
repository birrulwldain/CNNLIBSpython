import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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
    nist_wavelengths = nist_data["obs_wl_air(nm)"].values
    nist_intensities = nist_data["intens"].values
    return np.array(nist_wavelengths), np.array(nist_intensities)


# Fungsi untuk menghapus sinyal latar belakang
def remove_background(intensities, window_length=51, polyorder=3):
    background = savgol_filter(intensities, window_length, polyorder)
    corrected_intensities = intensities - background
    return background, corrected_intensities


# Fungsi untuk mengkalibrasi spektrum untuk atom spesifik (kalsium)
def calibrate_spectrum(
    measured_wavelengths, measured_intensities, nist_wavelengths, nist_intensities
):
    # Normalisasi intensitas terukur (ASC)
    normalized_measured_intensities = measured_intensities / np.max(
        measured_intensities
    )

    # Normalisasi intensitas NIST
    normalized_nist_intensities = nist_intensities / np.max(nist_intensities)

    # Sesuaikan panjang gelombang terukur dengan data NIST
    calibration_factors = []
    for wl in measured_wavelengths:
        closest_nist_wl = nist_wavelengths[np.argmin(np.abs(nist_wavelengths - wl))]
        calibration_factors.append(closest_nist_wl / wl)
    mean_calibration_factor = np.mean(calibration_factors)
    calibrated_wavelengths = measured_wavelengths * mean_calibration_factor

    # Interpolasi intensitas NIST agar sesuai dengan panjang gelombang terkalibrasi
    f = interp1d(
        nist_wavelengths,
        normalized_nist_intensities,
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_nist_intensities = f(calibrated_wavelengths)

    # Kalibrasi intensitas terukur dengan intensitas NIST yang diinterpolasi
    calibrated_intensities = (
        normalized_measured_intensities * interpolated_nist_intensities
    )

    return calibrated_wavelengths, calibrated_intensities


# Nama file input ASC dan NIST CSV
input_filename = "../expdata/spectrum_data.asc"
nist_filename = "../expdata/CaI-CaII.csv"

# Baca data dari file ASC dan NIST CSV
wavelengths, intensities = read_from_asc(input_filename)
nist_wavelengths, nist_intensities = read_nist_csv(nist_filename)

# Normalisasi intensitas LIBS ASC dan NIST
normalized_intensities = intensities / np.max(intensities)
normalized_nist_intensities = nist_intensities / np.max(nist_intensities)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(normalized_intensities)

# Deteksi puncak
height_threshold = 0.001 * np.max(corrected_intensities)
distance_between_peaks = 1
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Plot spektrum dengan puncak yang diidentifikasi
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, corrected_intensities, label="Spektrum Koreksi")
plt.plot(
    wavelengths[peaks], corrected_intensities[peaks], "x", label="Puncak Terdeteksi"
)
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas (Normalisasi)")
plt.legend()
plt.grid(True)
plt.title("Puncak Terdeteksi pada Spektrum LIBS")
plt.show()

# Kalibrasi spektrum untuk kalsium
calibrated_wavelengths, calibrated_intensities = calibrate_spectrum(
    wavelengths[peaks], corrected_intensities[peaks], nist_wavelengths, nist_intensities
)

# Plot spektrum yang telah dikalibrasi
plt.figure(figsize=(10, 6))
plt.plot(
    calibrated_wavelengths,
    calibrated_intensities,
    "o",
    label="Spektrum Terukur (Terkalibrasi)",
)
plt.plot(nist_wavelengths, normalized_nist_intensities, "x", label="data NIST")
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas (Normalisasi)")
plt.legend()
plt.grid(True)
plt.title("Spektrum LIBS Setelah Kalibrasi dengan data NIST (Kalsium)")
plt.show()

# Simpan hasil kalibrasi ke dalam file CSV
calibrated_data = pd.DataFrame(
    {
        "Wavelength (nm)": calibrated_wavelengths,
        "Intensity (Normalized)": calibrated_intensities,
    }
)
calibrated_data.to_csv("calibrated_spectrum.csv", index=False)
