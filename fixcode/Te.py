import numpy as np

from fixcode.checkpeaknist import corrected_intensities
from fixcode.checkpeaknist import peaks
from fixcode.checkpeaknist import wavelengths


def estimate_Te(wavelengths, intensities):
    # Misalnya, ambil garis-garis emisi yang relevan
    # Di sini, contoh sederhana dengan dua puncak garis emisi
    emission_peaks = wavelengths[peaks][:2]  # Ambil dua puncak pertama

    # Misalnya, hitung energi eksitasi untuk garis-garis ini (secara nyata dari data Anda)
    # Di sini, menggunakan nilai arbitrer untuk demonstrasi
    excitation_energies = np.array([3.0, 3.5])  # Misalnya, dalam eV

    # Plot Boltzmann
    inverse_energy = 1 / excitation_energies
    log_intensity = np.log(corrected_intensities[peaks][:2])

    # Hitung gradien untuk mendapatkan suhu Te
    gradient, intercept = np.polyfit(inverse_energy, log_intensity, 1)
    Te_estimate = -1 / gradient  # Perkiraan suhu dalam K

    return Te_estimate


# Panggil fungsi untuk mendapatkan estimasi suhu elektron
Te_estimate = estimate_Te(wavelengths, corrected_intensities)
print(f"Estimasi Suhu Elektron Plasma: {Te_estimate:.2f} K")
