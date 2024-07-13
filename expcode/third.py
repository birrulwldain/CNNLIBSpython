import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


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


# Fungsi untuk mendeteksi puncak
def detect_peaks(intensities, height=None, distance=None):
    peaks, _ = find_peaks(intensities, height=height, distance=distance)
    return peaks


def remove_background(intensities, window_length=51, polyorder=3):
    # Menghaluskan sinyal untuk memperkirakan latar belakang
    background = savgol_filter(intensities, window_length, polyorder)
    # Mengurangkan latar belakang dari sinyal asli
    corrected_intensities = intensities - background
    return background, corrected_intensities


def normalize(intensities):
    max_intensity = np.max(intensities)
    normalized_intensities = intensities / max_intensity
    return normalized_intensities


# Nama file input
input_filename = "../expdata/ba6.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Deteksi puncak dengan parameter yang sesuai
height_threshold = 0.1 * np.max(intensities)  # threshold tinggi untuk deteksi puncak
distance_between_peaks = 5  # jarak minimum antara puncak
background, corrected_intensities = remove_background(intensities)
normalized_intensities = normalize(corrected_intensities)
peaks = detect_peaks(
    intensities, height=height_threshold, distance=distance_between_peaks
)

# Membuat plot
plt.figure(figsize=(10, 6))

plt.plot(wavelengths, intensities, label="Spektrum Asli")
plt.plot(
    wavelengths, background, label="Latar Belakang yang Diperkirakan", linestyle="--"
)
plt.plot(wavelengths, corrected_intensities, label="Spektrum tanpa Latar Belakang")
plt.plot(
    wavelengths,
    normalized_intensities,
    label="Spektrum Ter-normalisasi",
    linestyle="-.",
)
plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.title("Penghapusan Sinyal Latar Belakang dari Spektrum LIBS")
plt.legend()
plt.grid(True)

# Menyimpan plot ke file PDF dengan resolusi tinggi
plt.savefig(
    "spectrum_corrected_normalized_high_res.pdf", format="pdf", dpi=300
)  # dpi menentukan resolusi
plt.show()

# Anotasi titik puncak
plt.figure(figsize=(10, 4))
plt.plot(wavelengths, intensities, label="Spektrum Asli")
plt.plot(
    wavelengths,
    normalized_intensities,
    label="Spektrum Ter-normalisasi",
    linestyle="-.",
)
for peak in peaks:
    plt.plot(
        wavelengths[peak], intensities[peak], "r"
    )  # Titik puncak dengan tanda 'x' merah
    plt.text(
        wavelengths[peak],
        intensities[peak],
        f"{wavelengths[peak]:.7f}",
        fontsize=6,
        ha="right",
        va="bottom",
        rotation=90,
    )

plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.title("Deteksi Puncak pada Spektrum LIBS")
plt.legend()

# Menyimpan plot ke file PDF dengan resolusi tinggi
plt.savefig(
    "spectrum_normalized_high_res_peaks.pdf", format="pdf", dpi=300
)  # dpi menentukan resolusi
plt.show()

print(
    "Gambar telah disimpan dalam format PDF dengan resolusi tinggi dan anotasi titik puncak."
)
for peak in peaks:
    print(
        f"Puncak terdeteksi pada panjang gelombang: {wavelengths[peak]} nm dengan intensitas: {intensities[peak]}"
    )
