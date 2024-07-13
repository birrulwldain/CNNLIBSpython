import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz  # Voigt profile


class FitVoight(object):
    @staticmethod
    def voigt(x, amp, cen, sigma, gamma):
        z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
        return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def read_asc_file(file_path):
        data = np.loadtxt(file_path, skiprows=1)
        wavelength = data[:, 0]
        intensity = data[:, 1]
        return wavelength, intensity

    @staticmethod
    def estimate_noise(intensity):
        noise = np.std(intensity)
        return noise

    @staticmethod
    def calculate_residuals(y, y_fit):
        return y - y_fit

    @staticmethod
    def calculate_chi_squared(y, y_fit, errors):
        return np.sum(((y - y_fit) / errors) ** 2)

    @staticmethod
    def fit_voigt_peak(wavelength, intensity, peak_index, window=5, maxfev=5000):
        start = max(0, peak_index - window)
        end = min(len(wavelength), peak_index + window + 1)
        x = wavelength[start:end]
        y = intensity[start:end]

        amp_init = max(y)
        cen_init = wavelength[peak_index]
        sigma_init = np.std(x)
        gamma_init = np.std(x)

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

    @staticmethod
    def fit_contoh_voigt(wavelength, intensity, peak_index, window=5, maxfev=5000):
        x_zoom, y_zoom, fit_voigt_curve_zoom, _ = FitVoight.fit_voigt_peak(
            wavelength, intensity, peak_index, window, maxfev
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            x_zoom, y_zoom, "o", label=f"data Puncak {wavelength[peak_index]:.2f} nm"
        )
        plt.plot(
            x_zoom,
            fit_voigt_curve_zoom,
            label=f"Voigt Fit {wavelength[peak_index]:.2f} nm",
            color="red",
        )
        plt.xlabel("Panjang Gelombang (nm)")
        plt.ylabel("Intensitas")
        plt.legend()
        plt.title(f"Zoom pada Puncak Tertinggi di {wavelength[peak_index]:.2f} nm")
        plt.show()


# File ASC yang akan dibaca
file_path = "data/GRUP 5_SAMPEL 5_D 0.2 us_skala 5_2.asc"

# Membaca data spektral dari file ASC
wavelength, intensity = FitVoight.read_asc_file(file_path)

# Estimasi noise
noise = FitVoight.estimate_noise(intensity)

# Temukan puncak-puncak dalam spektrum
peaks, _ = find_peaks(intensity, height=noise * 3)

# Lakukan fitting Voigt untuk setiap puncak yang ditemukan
plt.figure(figsize=(10, 6))
plt.plot(wavelength, intensity, label="Spektrum LIBS Asli", color="black")

for peak_index in peaks:
    x, y, fit_voigt_curve, _ = FitVoight.fit_voigt_peak(
        wavelength, intensity, peak_index
    )

    if fit_voigt_curve is not None:
        plt.plot(x, y, "o", label=f"data Puncak {wavelength[peak_index]:.2f} nm")
        plt.plot(x, fit_voigt_curve, label=f"Voigt Fit {wavelength[peak_index]:.2f} nm")

plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.title("Fitting Voigt untuk Setiap Puncak dengan S/N >= 3")
plt.show()

# Plot zoom pada puncak tertinggi
if peaks.size > 0:
    highest_peak_index = peaks[np.argmax(intensity[peaks])]
    FitVoight.fit_contoh_voigt(
        wavelength, intensity, highest_peak_index, window=5, maxfev=5000
    )
