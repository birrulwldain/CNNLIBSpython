import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, savgol_filter  # Tambahkan savgol_filter di sini


def get_spectrum_average(db_path, sample_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT wavelength, AVG(intensity) as avg_intensity
        FROM spectrum_data
        WHERE sample_name = ?
        GROUP BY wavelength
        ORDER BY wavelength
        """,
        (sample_name,),
    )
    data = cursor.fetchall()
    conn.close()

    if not data:
        print(f"Tidak ada data ditemukan untuk sampel: {sample_name}")
        return np.array([]), np.array([])

    wavelengths, average_intensities = zip(*data)
    return np.array(wavelengths), np.array(average_intensities)


def read_nist_csv(nist_filename):
    nist_data = pd.read_csv(nist_filename)

    def clean_string(s):
        return "".join(
            [
                char
                for char in str(s)
                if char is not None and (char.isdigit() or char == ".")
            ]
        )

    nist_data["obs_wl_air(nm)"] = (
        nist_data["obs_wl_air(nm)"]
        .apply(lambda x: str(x).replace('="', "0").replace('"', "0"))
        .apply(clean_string)
        .astype(float)
    )
    nist_data["intens"] = (
        nist_data["intens"]
        .apply(lambda x: str(x).replace('="', "0").replace('"', "0"))
        .apply(clean_string)
        .astype(float)
    )
    nist_data["sp_num"] = nist_data["sp_num"].astype(int)

    nist_wavelengths = nist_data["obs_wl_air(nm)"].values
    nist_element = nist_data["element"].values
    nist_num = nist_data["sp_num"].values

    return nist_wavelengths, nist_element, nist_num


def find_closest_peaks(
        wavelengths,
        intensities,
        nist_wavelengths,
        nist_element,
        nist_num,
        prominence=0.1,
        height=0.1,
        tolerance=10,
):
    peaks, _ = find_peaks(intensities, prominence=prominence, height=height)
    peak_wavelengths = wavelengths[peaks]
    peak_intensities = intensities[peaks]

    closest_peaks = []
    for i, peak_wl in enumerate(peak_wavelengths):
        distances = np.abs(nist_wavelengths - peak_wl)
        sorted_indices = np.argsort(distances)[:20]
        for idx in sorted_indices:
            if distances[idx] < tolerance and nist_num[idx] in [1, 2]:
                closest_peaks.append(
                    (
                        peak_wl,
                        peak_intensities[i],
                        nist_wavelengths[idx],
                        nist_element[idx],
                        nist_num[idx],
                        distances[idx],
                    )
                )
    return closest_peaks


def plot_spectra_segment(
        sample_name,
        wavelengths,
        intensities,
        segment_size,
        nist_wavelengths,
        nist_element,
        nist_num,
        pdf_filename,
        prominence,
        height,
        savgol_window=11,  # Tambahkan parameter untuk jendela Savitzky-Golay
        savgol_poly=3,  # Tambahkan parameter untuk orde polinomial Savitzky-Golay
):
    # Terapkan smoothing Savitzky-Golay
    intensities_smooth = savgol_filter(intensities, savgol_window, savgol_poly)

    num_segments = int(np.ceil((wavelengths[-1] - wavelengths[0]) / segment_size))
    pdf_pages = PdfPages(pdf_filename)

    # Plot full spectrum on the first page
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, intensities_smooth, color="black", linewidth=0.4)
    ax.set_xlabel("Panjang Gelombang (nm)")
    ax.set_ylabel("Intensitas")
    ax.set_title(f"Spektrum Penuh Sampel {sample_name}")
    pdf_pages.savefig(fig)
    plt.close(fig)

    # Plot segmented spectra on subsequent pages
    for i in range(num_segments):
        segment_start = wavelengths[0] + i * segment_size
        segment_end = segment_start + segment_size

        mask = (wavelengths >= segment_start) & (wavelengths <= segment_end)
        wavelengths_zoomed = wavelengths[mask]
        intensities_zoomed = intensities_smooth[mask]

        if len(wavelengths_zoomed) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelengths_zoomed, intensities_zoomed, color="black", linewidth=0.4)

        closest_peaks = find_closest_peaks(
            wavelengths_zoomed,
            intensities_zoomed,
            nist_wavelengths,
            nist_element,
            nist_num,
            prominence=prominence,
            height=height,
        )
        selected_peaks = []
        for j, (peak_wl, peak_int, nist_wl, element, ion_stage, distance) in enumerate(
                closest_peaks
        ):
            print(
                f"{j + 1}: {element} {ion_stage} @ {nist_wl:.6f} nm (Δλ = {distance:.6f} nm)"
            )
            if (j + 1) % 20 == 0 or j == len(closest_peaks) - 1:
                choice = input(
                    f"Pilih puncak untuk {peak_wl:.6f} nm (masukkan nomor atau tekan Enter untuk lewati): "
                )
                if choice.isdigit():
                    selected_idx = int(choice) - 1
                    if 0 <= selected_idx < len(closest_peaks):
                        selected_peaks.append(closest_peaks[selected_idx])
                    else:
                        print("Pilihan tidak valid, puncak otomatis akan dipilih.")
                        selected_peaks.append(
                            min(closest_peaks[j - 19: j + 1], key=lambda x: x[-1])
                        )
                else:
                    selected_peaks.append(
                        min(closest_peaks[j - 19: j + 1], key=lambda x: x[-1])
                    )
                    print(
                        f"Puncak otomatis dipilih: {min(closest_peaks[j - 19:j + 1], key=lambda x: x[-1])[2]:.6f} nm"
                    )

                if (j + 1) % 20 == 0:
                    print("Tampilkan 20 puncak terdekat berikutnya...")

        for peak_wl, peak_int, nist_wl, element, ion_stage, _ in selected_peaks:
            ion_stage = (
                "I" if ion_stage == 1 else "II" if ion_stage == 2 else str(ion_stage)
            )
            ax.scatter(peak_wl, peak_int, color="red", s=8, marker=".")
            label = f"{element} {ion_stage} {nist_wl:.6f} nm"
            ax.text(
                peak_wl,
                peak_int,
                label,
                fontsize=6,
                ha="center",
                va="bottom",
                rotation=90,
                color="blue",
            )

        ax.set_xlabel("Panjang Gelombang (nm)")
        ax.set_ylabel("Intensitas")
        ax.set_title(
            f"Spektrum Sampel {sample_name} pada {segment_start:.2f}-{segment_end:.2f} nm"
        )
        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()
    print(f"Spektrum tersimpan dalam {pdf_filename}")


def main():
    db_path = "tanah_vulkanik1.db"
    nist_filename = "CaI-CaII.csv"

    nist_wavelengths, nist_element, nist_num = read_nist_csv(nist_filename)

    while True:
        sample_name = f'TV{input("Sampel: ")}'

        wavelengths, average_intensities = get_spectrum_average(db_path, sample_name)

        if len(wavelengths) > 0 and len(average_intensities) > 0:
            segment_size = float(input("Masukkan ukuran segmen (nm): "))

            prominence = float(
                input("Masukkan nilai prominence (default: 0.1): ") or 0.1
            )
            height = float(input("Masukkan nilai height (default: 0.1): ") or 0.1)

            savgol_window = int(
                input("Masukkan ukuran jendela Savitzky-Golay (default: 11): ") or 11
            )
            savgol_poly = int(
                input("Masukkan orde polinomial Savitzky-Golay (default: 3): ") or 3
            )

            pdf_filename = f"{sample_name}_{segment_size}({height};{prominence}).pdf"
            plot_spectra_segment(
                sample_name,
                wavelengths,
                average_intensities,
                segment_size,
                nist_wavelengths,
                nist_element,
                nist_num,
                pdf_filename,
                prominence,
                height,
                savgol_window,
                savgol_poly,
            )
        else:
            print("Data spektrum tidak ditemukan untuk kombinasi yang diberikan.")

        if input("Apakah Anda ingin keluar? (y/n): ").strip().lower() == "y":
            break


if __name__ == "__main__":
    main()
