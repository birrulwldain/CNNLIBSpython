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
    # Menghaluskan sinyal untuk memperkirakan latar belakang
    background = savgol_filter(intensities, window_length, polyorder)
    # Mengurangkan latar belakang dari sinyal asli
    corrected_intensities = intensities - background
    return background, corrected_intensities


# Fungsi untuk mendeteksi puncak
def detect_peaks(intensities, height=None, distance=None):
    peaks, _ = find_peaks(intensities, height=height, distance=distance)
    return peaks


# Nama file input
input_filename = "../expdata/ba6.asc"

# Baca data dari file ASC
wavelengths, intensities = read_from_asc(input_filename)

# Hapus sinyal latar belakang
background, corrected_intensities = remove_background(intensities)

# Deteksi puncak dengan parameter yang sesuai
height_threshold = 0.1 * np.max(
    corrected_intensities
)  # threshold tinggi untuk deteksi puncak
distance_between_peaks = 5  # jarak minimum antara puncak

peaks = detect_peaks(
    corrected_intensities, height=height_threshold, distance=distance_between_peaks
)

# Membuat plot untuk setiap puncak dengan zoom
zoom_window = 10  # Jarak panjang gelombang sekitar puncak untuk di-zoom

for peak in peaks:
    start = max(0, peak - zoom_window)
    end = min(len(wavelengths), peak + zoom_window)

    plt.figure(figsize=(6, 4))  # Ukuran figur
    plt.plot(
        wavelengths[start:end],
        corrected_intensities[start:end],
        label="Spektrum tanpa Latar Belakang",
    )
    plt.plot(
        wavelengths[peak], corrected_intensities[peak], "rx"
    )  # Titik puncak dengan tanda 'x' merah
    plt.xlabel("Panjang Gelombang (nm)")
    plt.ylabel("Intensitas")
    plt.title(f"Puncak pada Panjang Gelombang {wavelengths[peak]:.2f} nm")
    plt.legend()
    plt.grid(True)

    # Menyimpan plot ke file PDF per puncak
    plt.savefig(
        f"peak_{wavelengths[peak]:.2f}_nm_zoomed.pdf", format="pdf", dpi=300
    )  # dpi menentukan resolusi
    plt.close()  # Menutup plot setelah menyimpan

print(
    "Gambar untuk setiap puncak telah disimpan dalam format PDF dengan resolusi tinggi dan zoom pada puncak."
)
