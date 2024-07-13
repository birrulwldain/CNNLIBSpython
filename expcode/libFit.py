# Fungsi Lorentzian
def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x - cen) ** 2 + wid**2)


def read_asc_file(file_path):
    data = np.loadtxt(
        file_path, skiprows=1
    )  # Membaca file ASC, melewati baris pertama jika perlu
    wavelength = data[:, 0]
    intensity = data[:, 1]
    return wavelength, intensity


def estimate_noise(intensity):
    noise = np.std(intensity)
    return noise


# Fungsi untuk melakukan fitting Lorentzian pada rentang tertentu
def fit_lorentzian_peak(wavelength, intensity, peak_index, window=5, maxfev=5000):
    # Tentukan rentang di sekitar puncak
    start = max(0, peak_index - window)
    end = min(len(wavelength), peak_index + window + 1)

    x = wavelength[start:end]
    y = intensity[start:end]

    # Inisialisasi parameter fitting
    amp_init = max(y)
    cen_init = wavelength[peak_index]
    wid_init = np.std(x)

    # Melakukan fitting dengan Lorentzian
    try:
        popt_lorentzian, _ = curve_fit(
            lorentzian,
            x,
            y,
            p0=[amp_init, cen_init, wid_init],
            bounds=([0, min(x), 0], [np.inf, max(x), np.inf]),
            maxfev=maxfev,
        )
    except RuntimeError as e:
        print(f"Fitting gagal untuk puncak di {cen_init:.2f} nm: {e}")
        return x, y, None, None

    fit_lorentzian_curve = lorentzian(x, *popt_lorentzian)

    return x, y, fit_lorentzian_curve, popt_lorentzian


# Fungsi Gaussian
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


def read_asc_file(file_path):
    data = np.loadtxt(
        file_path, skiprows=1
    )  # Membaca file ASC, melewati baris pertama jika perlu
    wavelength = data[:, 0]
    intensity = data[:, 1]
    return wavelength, intensity


def estimate_noise(intensity):
    noise = np.std(intensity)
    return noise


# Fungsi untuk melakukan fitting pada rentang tertentu
def fit_gaussian_peak(wavelength, intensity, peak_index, window=5, maxfev=5000):
    # Tentukan rentang di sekitar puncak
    start = max(0, peak_index - window)
    end = min(len(wavelength), peak_index + window + 1)

    x = wavelength[start:end]
    y = intensity[start:end]

    # Inisialisasi parameter fitting
    amp_init = max(y)
    cen_init = wavelength[peak_index]
    wid_init = np.std(x)

    # Melakukan fitting dengan Gaussian
    try:
        popt_gaussian, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=[amp_init, cen_init, wid_init],
            bounds=([0, min(x), 0], [np.inf, max(x), np.inf]),
            maxfev=maxfev,
        )
    except RuntimeError as e:
        print(f"Fitting gagal untuk puncak di {cen_init:.2f} nm: {e}")
        return x, y, None, None

    fit_gaussian_curve = gaussian(x, *popt_gaussian)

    return x, y, fit_gaussian_curve, popt_gaussian


import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz  # Voigt profile


# Fungsi Voigt
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
            voigt,
            x,
            y,
            p0=[amp_init, cen_init, sigma_init, gamma_init],
            bounds=([0, min(x), 0, 0], [np.inf, max(x), np.inf, np.inf]),
            maxfev=maxfev,
        )
    except RuntimeError as e:
        print(f"Fitting gagal untuk puncak di {cen_init:.2f} nm: {e}")
        return x, y, None, None

    fit_voigt_curve = voigt(x, *popt_voigt)

    return x, y, fit_voigt_curve, popt_voigt
