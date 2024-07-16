import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from fixcode.FitGaussian import FitGaussian
from fixcode.FitLorentz import FitLorentz
from fixcode.FitVoight import FitVoight


# Fungsi untuk membaca file ASC
def read_asc_file(dataasc):
    file_path = f"data/{dataasc}"
    data = np.loadtxt(file_path, skiprows=0)
    wavelength = data[:, 0]
    intensity = data[:, 1]
    return wavelength, intensity


# Fungsi untuk menghitung noise
def estimate_noise(intensity):
    noise = np.std(intensity)
    return noise


dataasc = "Cu plate_skala 5_D 1 us_2.asc"

# Membaca data spektral dari file ASC
wavelength, intensity = read_asc_file(dataasc)

# Estimasi noise
noise = estimate_noise(intensity)

# Temukan puncak-puncak dalam spektrum
peaks, properties = find_peaks(intensity, height=noise * 3)

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


# Fungsi untuk menghitung residuals
def calculate_residuals(y, y_fit):
    return y - y_fit


# Fungsi untuk menghitung chi-squared
def calculate_chi_squared(y, y_fit, errors):
    return np.sum(((y - y_fit) / errors) ** 2)


# Evaluasi hasil fitting untuk setiap model
for peak_index in peaks:
    x_gaussian, y_gaussian, fit_gaussian_curve, popt_gaussian = (
        FitGaussian.fit_gaussian_peak(wavelength, intensity, peak_index)
    )
    x_lorentzian, y_lorentzian, fit_lorentzian_curve, popt_lorentzian = (
        FitLorentz.fit_lorentzian_peak(wavelength, intensity, peak_index)
    )
    x_voigt, y_voigt, fit_voigt_curve, popt_voigt = FitVoight.fit_voigt_peak(
        wavelength, intensity, peak_index
    )

    if fit_gaussian_curve is not None:
        residuals_gaussian = calculate_residuals(y_gaussian, fit_gaussian_curve)
        chi_squared_gaussian = calculate_chi_squared(
            y_gaussian, fit_gaussian_curve, np.sqrt(y_gaussian)
        )
        print(f"Gaussian Fit untuk puncak di {wavelength[peak_index]:.2f} nm:")
        print(f"  Parameter: {popt_gaussian}")
        print(f"  Residuals: {residuals_gaussian}")
        print(f"  Chi-Squared: {chi_squared_gaussian}")

    if fit_lorentzian_curve is not None:
        residuals_lorentzian = calculate_residuals(y_lorentzian, fit_lorentzian_curve)
        chi_squared_lorentzian = calculate_chi_squared(
            y_lorentzian, fit_lorentzian_curve, np.sqrt(y_lorentzian)
        )
        print(f"Lorentzian Fit untuk puncak di {wavelength[peak_index]:.2f} nm:")
        print(f"  Parameter: {popt_lorentzian}")
        print(f"  Residuals: {residuals_lorentzian}")
        print(f"  Chi-Squared: {chi_squared_lorentzian}")

    if fit_voigt_curve is not None:
        residuals_voigt = calculate_residuals(y_voigt, fit_voigt_curve)
        chi_squared_voigt = calculate_chi_squared(
            y_voigt, fit_voigt_curve, np.sqrt(y_voigt)
        )
        print(f"Voigt Fit untuk puncak di {wavelength[peak_index]:.2f} nm:")
        print(f"  Parameter: {popt_voigt}")
        print(f"  Residuals: {residuals_voigt}")
        print(f"  Chi-Squared: {chi_squared_voigt}")

# Plot residuals untuk Voigt Fit pada puncak tertinggi
if highest_peak_index != -1:
    x_voigt, y_voigt, fit_voigt_curve, _ = FitVoight.fit_voigt_peak(
        wavelength, intensity, highest_peak_index
    )
    residuals_voigt = calculate_residuals(y_voigt, fit_voigt_curve)

    plt.figure(figsize=(10, 6))
    plt.plot(x_voigt, residuals_voigt, "o-", label="Residuals Voigt Fit", color="blue")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Panjang Gelombang (nm)")
    plt.ylabel("Residuals")
    plt.legend()
    plt.title(
        f"Residuals dari Voigt Fit pada Puncak Tertinggi di {wavelength[highest_peak_index]:.2f} nm"
    )
    plt.show()

# Plot residuals untuk Lorentzian Fit pada puncak tertinggi
if highest_peak_index != -1:
    x_lorentzian, y_lorentzian, fit_lorentzian_curve, _ = (
        FitLorentz.fit_lorentzian_peak(wavelength, intensity, highest_peak_index)
    )
    residuals_lorentzian = calculate_residuals(y_lorentzian, fit_lorentzian_curve)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_lorentzian,
        residuals_lorentzian,
        "o-",
        label="Residuals Lorentzian Fit",
        color="blue",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Panjang Gelombang (nm)")
    plt.ylabel("Residuals")
    plt.legend()
    plt.title(
        f"Residuals dari Lorentzian Fit pada Puncak Tertinggi di {wavelength[highest_peak_index]:.2f} nm"
    )
    plt.show()

# Plot residuals untuk Gaussian Fit pada puncak tertinggi
if highest_peak_index != -1:
    x_gaussian, y_gaussian, fit_gaussian_curve, _ = FitGaussian.fit_gaussian_peak(
        wavelength, intensity, highest_peak_index
    )
    residuals_gaussian = calculate_residuals(y_gaussian, fit_gaussian_curve)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_gaussian,
        residuals_gaussian,
        "o-",
        label="Residuals Gaussian Fit",
        color="blue",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Panjang Gelombang (nm)")
    plt.ylabel("Residuals")
    plt.legend()
    plt.title(
        f"Residuals dari Gaussian Fit pada Puncak Tertinggi di {wavelength[highest_peak_index]:.2f} nm"
    )
    plt.show()
