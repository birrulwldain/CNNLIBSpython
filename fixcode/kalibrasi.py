# Misalkan Anda sudah memiliki libs_peaks_wavelengths (posisi puncak dalam spektrum LIBS) dan matched_peaks (hasil pencocokan dengan data NIST)

# Hitung perbedaan antara posisi puncak LIBS dan NIST
shifts = [(libs_wl - nist_wl) for libs_wl, nist_wl in matched_peaks]

# Metode 1: Menggeser posisi puncak dalam spektrum LIBS
calibrated_peaks1 = [
    libs_wl - shift for libs_wl, shift in zip(libs_peaks_wavelengths, shifts)
]

# Metode 2: Menghitung faktor skala untuk kalibrasi
calibrated_peaks2 = [
    libs_wl * (nist_wl / libs_wl) for libs_wl, nist_wl in matched_peaks
]

# Contoh cetak hasil kalibrasi
print("Hasil Kalibrasi Posisi Puncak (Metode 1):")
for original, calibrated1 in zip(libs_peaks_wavelengths, calibrated_peaks1):
    print(f"Posisi Awal: {original:.2f} nm, Setelah Kalibrasi: {calibrated1:.2f} nm")

print("\nHasil Kalibrasi Posisi Puncak (Metode 2):")
for original, calibrated2 in zip(libs_peaks_wavelengths, calibrated_peaks2):
    print(f"Posisi Awal: {original:.2f} nm, Setelah Kalibrasi: {calibrated2:.2f} nm")
