import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz  # Voigt profile


class FitVoight(object):
    def voigt(x, amp, cen, sigma, gamma):
        z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
        return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    # Fungsi untuk membaca file ASC
    def read_asc_file(file_path):
        data = np.loadtxt(
            file_path, skiprows=1
        )  # Membaca file ASC, melewati baris pertama jika perlu
        wavelength = data[:, 0]
        intensity = data[:, 1]
        return wavelength, intensity

    # Fungsi untuk menghitung noise
    def estimate_noise(intensity):
        noise = np.std(intensity)
        return noise

    # Fungsi untuk melakukan fitting pada rentang tertentu
    def fit_voigt_peak(wavelength, intensity, peak_index, window=5, maxfev=5000):
        # Tentukan rentang di sekitar puncak
        start = max(0, peak_index - window)
        end = min(len(wavelength), peak_index + window + 1)

        x = wavelength[start:end]
        y = intensity[start:end]

        # Inisialisasi parameter fitting
        amp_init = max(y)
        cen_init = wavelength[peak_index]
        sigma_init = np.std(x)
        gamma_init = np.std(x)

        # Melakukan fitting dengan Voigt
        try:
            popt_voigt, _ = curve_fit(
                FitVoight.voigt,
                x,
                y,
                p0=[amp_init, cen_init, sigma_init, gamma_init],
                bounds=([0, min(x), 0, 0], [np.inf, max(x), np.inf, np.inf]),
                maxfev=maxfev,
            )
        except RuntimeError as e:
            print(f"Fitting gagal untuk puncak di {cen_init:.2f} nm: {e}")
            return x, y, None, None

        fit_voigt_curve = FitVoight.voigt(x, *popt_voigt)

        return x, y, fit_voigt_curve, popt_voigt

    def fit_contoh_voigt(wavelength, intensity, peak_index, window=5, maxfev=5000):
        if highest_peak_index != -1:
            x_zoom, y_zoom, fit_voigt_curve_zoom, _ = FitVoight.fit_voigt_peak(
                wavelength, intensity, highest_peak_index
            )

            plt.figure(figsize=(10, 6))
            plt.plot(
                x_zoom,
                y_zoom,
                "o",
                label=f"data Puncak {wavelength[highest_peak_index]:.2f} nm",
            )
            plt.plot(
                x_zoom,
                fit_voigt_curve_zoom,
                label=f"Voigt Fit {wavelength[highest_peak_index]:.2f} nm",
                color="red",
            )
            plt.xlabel("Panjang Gelombang (nm)")
            plt.ylabel("Intensitas")
            plt.legend()
            plt.title(
                f"Zoom pada Puncak Tertinggi di {wavelength[highest_peak_index]:.2f} nm"
            )
            plt.xlim(min(x_zoom), max(x_zoom))  # Batasi rentang sumbu x pada plot
            plt.ylim(min(y_zoom), max(y_zoom))  # Batasi rentang sumbu y pada plot
            plt.show()


# File ASC yang akan dibaca
file_path = "data/GRUP 5_SAMPEL 5_D 0.2 us_skala 5_2.asc"


# Membaca data spektral dari file ASC
wavelength, intensity = FitVoight.read_asc_file(file_path)

# Estimasi noise
noise = FitVoight.estimate_noise(intensity)

# Temukan puncak-puncak dalam spektrum
peaks, properties = find_peaks(
    intensity, height=noise * 3
)  # Hanya puncak dengan S/N >= 3

# Inisialisasi variabel untuk menyimpan puncak tertinggi
highest_peak_intensity = 0
highest_peak_index = -1

plt.figure(figsize=(10, 6))
plt.plot(wavelength, intensity, label="Spektrum LIBS Asli", color="black")

# Lakukan fitting Voigt untuk setiap puncak yang ditemukan
for peak_index in peaks:
    x, y, fit_voigt_curve, popt_voigt = FitVoight.fit_voigt_peak(
        wavelength, intensity, peak_index
    )

    if fit_voigt_curve is not None:
        plt.plot(x, y, "o", label=f"data Puncak {wavelength[peak_index]:.2f} nm")
        plt.plot(x, fit_voigt_curve, label=f"Voigt Fit {wavelength[peak_index]:.2f} nm")

        # Perbarui puncak tertinggi jika intensitas lebih tinggi ditemukan
        if max(y) > highest_peak_intensity:
            highest_peak_intensity = max(y)
            highest_peak_index = peak_index

plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.title("Fitting Voigt untuk Setiap Puncak dengan S/N >= 3")
plt.show()

# Plot zoom pada puncak tertinggi


FitVoight.fit_contoh_voigt(wavelength, intensity, peak_index, window=5, maxfev=5000)
