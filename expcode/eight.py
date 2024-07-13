import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress


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


# Fungsi untuk mengukur intensitas puncak
def measure_intensity(wavelengths, intensities, peak_index):
    return intensities[peak_index]


# data tingkat energi dan faktor partisi untuk Ca II
levels = [
    {"wavelength": 393.37, "E": 3.15, "g": 6},
    {"wavelength": 396.85, "E": 3.12, "g": 4},
    {"wavelength": 422.67, "E": 3.20, "g": 4},
    {"wavelength": 428.30, "E": 3.23, "g": 6},
]

# Konstanta
k_B = 8.617333262145e-5  # eV/K

# Nama file input
input_filename = "../expdata/spectrum_data.asc"

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

# Identifikasi puncak yang sesuai dengan spektrum Ca
identified_peaks = []
for level in levels:
    ca_wl = level["wavelength"]
    peak_index = np.argmin(np.abs(wavelengths[peaks] - ca_wl))
    if np.abs(wavelengths[peaks][peak_index] - ca_wl) < 1:  # Toleransi kesesuaian
        identified_peaks.append(peaks[peak_index])
        level["intensity"] = corrected_intensities[peaks[peak_index]]

# Hitung fungsi Boltzmann
ln_I_over_g = []
inverse_E = []
for level in levels:
    if "intensity" in level:
        I = level["intensity"]
        g = level["g"]
        E = level["E"]
        if I > 0:
            ln_I_over_g.append(np.log(I / g))
            inverse_E.append(1 / (k_B * E))

# Periksa apakah ada cukup data untuk melakukan regresi linier
if len(ln_I_over_g) < 2:
    raise ValueError(
        "Tidak cukup data untuk melakukan regresi linier. Pastikan data intensitas valid dan memiliki variasi dalam nilai energi."
    )

# Plot Boltzmann
ln_I_over_g = np.array(ln_I_over_g)
inverse_E = np.array(inverse_E)
slope, intercept, r_value, p_value, std_err = linregress(inverse_E, ln_I_over_g)
T_e = -1 / slope

plt.figure(figsize=(8, 6))
plt.plot(inverse_E, ln_I_over_g, "o", label="data")
plt.plot(inverse_E, slope * inverse_E + intercept, "-", label=f"Fit: T_e = {T_e:.2f} K")
plt.xlabel("1 / (k_B * E) (1/eV)")
plt.ylabel("ln(I/g)")
plt.legend()
plt.grid(True)
plt.title("Boltzmann Plot untuk Penghitungan Suhu Plasma")
plt.savefig("boltzmann_plot_ca.pdf", format="pdf", dpi=300)
plt.show()

print(f"Suhu plasma yang dihitung: {T_e:.2f} K")

# Hitung Densitas Elektron menggunakan Pelebaran Stark
# Asumsikan bahwa kita memiliki informasi pelebaran Stark untuk garis 422.67 nm
# Delta_lambda (peleburan Stark) diukur dalam nm
Delta_lambda = 0.05  # FWHM

# Konstanta untuk Stark broadening
e = 1.602176634e-19  # Muatan elektron dalam Coulomb
epsilon_0 = 8.854187817e-12  # Permitivitasi vakum dalam F/m
m_e = 9.10938356e-31  # Massa elektron dalam kg
c = 3e8  # Kecepatan cahaya dalam m/s
h = 6.62607015e-34  # J·s, konstanta Planck
c = 3.0e8  # m/s, kecepatan cahaya
# Perhitungan Densitas Elektron
densitas_elektron = (Delta_lambda / 0.1) ** (1 / 1.2) * 1e17  # cm^-3
densitas_elektron = densitas_elektron * 1e6  # m^-3

print(f"Densitas Elektron yang dihitung: {densitas_elektron:.2e} m^-3")

# Hitung Konsentrasi Ca (Metode Simplifikasi)
# Asumsikan kita memiliki nilai A_ki (probabilitas transisi) dan n_e (densitas elektron)
A_ki = 1e8  # Nilai contoh, harus disesuaikan dengan data eksperimen
g_k = 4  # Faktor degenerasi tingkat atas
E_i = 2.93  # Energi ionisasi dalam eV
n_e = densitas_elektron

# Konsentrasi Ca
konsentrasi_Ca = (4 * np.pi / (g_k * A_ki)) * (I / (h * c)) * np.exp(E_i / (k_B * T_e))

print(f"Konsentrasi Ca yang dihitung: {konsentrasi_Ca:.2e} m^-3")
