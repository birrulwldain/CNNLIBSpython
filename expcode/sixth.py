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


# Fungsi untuk mendeteksi puncak dan mengukur FWHM
def measure_fwhm(wavelengths, intensities, peak_index):
    half_max = intensities[peak_index] / 2.0
    left_idx = np.where(intensities[:peak_index] <= half_max)[0]
    right_idx = np.where(intensities[peak_index:] <= half_max)[0] + peak_index

    if len(left_idx) == 0 or len(right_idx) == 0:
        return None  # Tidak dapat menghitung FWHM

    left_intercept = wavelengths[left_idx[-1]]
    right_intercept = wavelengths[right_idx[0]]
    fwhm = right_intercept - left_intercept
    return fwhm


# Fungsi untuk menghitung densitas elektron dari FWHM
def calculate_electron_density(fwhm, ion_species):
    # Gunakan koefisien Stark yang sesuai untuk ion_species
    # Misal: fwhm_nm * A / B = densitas elektron dalam satuan tertentu
    # Contoh koefisien fiktif:
    A = 1e16  # faktor skala fiktif
    B = 1.0  # faktor skala fiktif
    electron_density = (fwhm * A) / B
    return electron_density


# Nama file input
input_filename = "../expdata/ba6.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Deteksi puncak
height_threshold = 0.1 * np.max(corrected_intensities)
distance_between_peaks = 5
peaks, _ = find_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Asumsi kita tahu ion_species untuk puncak tertentu
ion_species = "H_alpha"  # Contoh spesies ion

# Hitung densitas elektron untuk setiap puncak yang terdeteksi
for peak in peaks:
    fwhm = measure_fwhm(wavelengths, corrected_intensities, peak)
    if fwhm is not None:
        electron_density = calculate_electron_density(fwhm, ion_species)
        plt.figure(figsize=(6, 4))
        start = max(0, peak - 10)
        end = min(len(wavelengths), peak + 10)
        plt.plot(
            wavelengths[start:end],
            corrected_intensities[start:end],
            label="Spektrum tanpa Latar Belakang",
        )
        plt.plot(wavelengths[peak], corrected_intensities[peak], "rx")
        plt.xlabel("Panjang Gelombang (nm)")
        plt.ylabel("Intensitas")
        plt.title(
            f"Puncak pada Panjang Gelombang {wavelengths[peak]:.2f} nm\nDensitas Elektron: {electron_density:.2e} cm^-3"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"peak_{wavelengths[peak]:.2f}_nm_electron_density.pdf",
            format="pdf",
            dpi=300,
        )
        plt.close()

print("Densitas elektron telah dihitung dan gambar telah disimpan dalam format PDF.")
