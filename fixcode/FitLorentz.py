import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


class FitLorentz(object):
    @staticmethod
    def lorentzian(x, amp, cen, wid):
        return amp * wid**2 / ((x - cen) ** 2 + wid**2)

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
    def fit_lorentzian_peak(wavelength, intensity, peak_index, window=5, maxfev=5000):
        start = max(0, peak_index - window)
        end = min(len(wavelength), peak_index + window + 1)
        x = wavelength[start:end]
        y = intensity[start:end]

        amp_init = max(y)
        cen_init = wavelength[peak_index]
        wid_init = np.std(x)

        try:
            popt_lorentzian, _ = curve_fit(
                FitLorentz.lorentzian,
                x,
                y,
                p0=[amp_init, cen_init, wid_init],
                bounds=([0, min(x), 0], [np.inf, max(x), np.inf]),
                maxfev=maxfev,
            )
        except RuntimeError as e:
            print(f"Fitting gagal untuk puncak di {cen_init:.2f} nm: {e}")
            return x, y, None, None

        fit_lorentzian_curve = FitLorentz.lorentzian(x, *popt_lorentzian)
        return x, y, fit_lorentzian_curve, popt_lorentzian

    @staticmethod
    def fit_contoh_lorentzian(wavelength, intensity, peak_index, window=5, maxfev=5000):
        x_zoom, y_zoom, fit_lorentzian_curve_zoom, _ = FitLorentz.fit_lorentzian_peak(
            wavelength, intensity, peak_index, window, maxfev
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            x_zoom,
            y_zoom,
            "o",
            label=f"data Puncak {wavelength[peak_index]:.2f} nm",
        )
        plt.plot(
            x_zoom,
            fit_lorentzian_curve_zoom,
            label=f"Lorentzian Fit {wavelength[peak_index]:.2f} nm",
            color="red",
        )
        plt.xlabel("Panjang Gelombang (nm)")
        plt.ylabel("Intensitas")
        plt.legend()
        plt.title(f"Zoom pada Puncak Tertinggi di {wavelength[peak_index]:.2f} nm")
        plt.xlim(min(x_zoom), max(x_zoom))  # Batasi rentang sumbu x pada plot
        plt.ylim(min(y_zoom), max(y_zoom))  # Batasi rentang sumbu y pada plot
        plt.show()


# File ASC yang akan dibaca
file_path = "data/Cu plate_skala 5_D 1 us_1.asc"

# Membaca data spektral dari file ASC
wavelength, intensity = FitLorentz.read_asc_file(file_path)

# Estimasi noise
noise = FitLorentz.estimate_noise(intensity)

# Temukan puncak-puncak dalam spektrum
peaks, _ = find_peaks(intensity, height=noise * 3)

# Inisialisasi variabel untuk menyimpan puncak tertinggi
highest_peak_intensity = 0
highest_peak_index = -1

plt.figure(figsize=(10, 6))
plt.plot(wavelength, intensity, label="Spektrum LIBS Asli", color="black")

# Lakukan fitting Lorentzian untuk setiap puncak yang ditemukan
for peak_index in peaks:
    x, y, fit_lorentzian_curve, _ = FitLorentz.fit_lorentzian_peak(
        wavelength, intensity, peak_index
    )

    if fit_lorentzian_curve is not None:
        plt.plot(x, y, "o", label=f"data Puncak {wavelength[peak_index]:.2f} nm")
        plt.plot(
            x,
            fit_lorentzian_curve,
            label=f"Lorentzian Fit {wavelength[peak_index]:.2f} nm",
            color="red",
        )

        # Perbarui puncak tertinggi jika intensitas lebih tinggi ditemukan
        if max(y) > highest_peak_intensity:
            highest_peak_intensity = max(y)
            highest_peak_index = peak_index

plt.xlabel("Panjang Gelombang (nm)")
plt.ylabel("Intensitas")
plt.legend()
plt.title("Fitting Lorentzian untuk Setiap Puncak dengan S/N >= 3")
plt.show()

# Plot zoom pada puncak tertinggi
if peaks.size > 0:
    highest_peak_index = peaks[np.argmax(intensity[peaks])]
    FitLorentz.fit_contoh_lorentzian(
        wavelength, intensity, highest_peak_index, window=5, maxfev=5000
    )
