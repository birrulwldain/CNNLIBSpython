import numpy as np

from fixcode.checkpeaknist import corrected_intensities
from fixcode.checkpeaknist import peaks
from fixcode.checkpeaknist import wavelengths


def estimate_ne(intensities, wavelengths):
    # Misalnya, hitung rasio intensitas garis emisi terhadap intensitas kontinu pada rentang tertentu
    # Di sini, contoh sederhana menggunakan rasio dua puncak yang mungkin merupakan garis emisi dan kontinu

    # Anda dapat menyesuaikan cara ini dengan metode yang lebih sesuai untuk data Anda
    peak_emission = intensities[peaks[0]]  # Intensitas puncak garis emisi
    peak_continuum = np.mean(
        intensities[peaks[0] - 10 : peaks[0] + 10]
    )  # Intensitas kontinu di sekitar puncak

    # Misalnya, hitung densitas elektron dengan metode sederhana
    ne = 1e17 * (
        peak_emission / peak_continuum
    )  # Contoh persamaan perbandingan sederhana

    return ne


# Panggil fungsi untuk mendapatkan densitas elektron
ne_estimate = estimate_ne(corrected_intensities, wavelengths)
print(f"Estimasi Densitas Elektron: {ne_estimate:.2e} cm^-3")
