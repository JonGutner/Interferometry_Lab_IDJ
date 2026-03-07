import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from Interferometry_Lab_IDJ import apply_global_calibration_Ivan as agci


def read_envelope_data():

    print("Name of envelope file: ")
    name = str(input())

    df = pd.read_csv("data/" + name + ".csv", skiprows=1)

    position = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values

    wavelength = position * 1e-7
    wavelength_nm = wavelength * 1e9

    return wavelength_nm, intensity


def get_grating_data():

    print("Name of grating file: ")
    name = str(input())

    _, wl, intensity = agci.data_g(name)

    wl_nm = wl * 1e9

    return wl_nm, intensity

def get_beam_splitter_ratio():
    bm_1 = "White_LED_Lens_SplitterRT"
    bm_2 = "White_LED_Lens_SplitterST"

    _, wl_1, int_1 = agci.data_g(bm_1)
    _, wl_2, int_2 = agci.data_g(bm_2)

    # 1. Interpolate int_2 onto wl_1 grid
    interp_func = interp1d(wl_2, int_2, bounds_error=False, fill_value=np.nan)
    int_2_interp = interp_func(wl_1)

    # 2. Create a mask: only keep data where the intensity is > 5% of max
    # This prevents division by near-zero noise
    threshold = 0.05 * np.nanmax(int_2_interp)
    mask = (int_2_interp > threshold) & (~np.isnan(int_2_interp))

    # 3. Apply mask and calculate ratio
    wl_clean = wl_1[mask]
    ratio_clean = int_1[mask] / int_2_interp[mask]

    # 4. Optional: Smooth the result to see the trend clearly
    ratio_smooth = savgol_filter(ratio_clean, 21, 3)

    plt.figure(figsize=(8, 5))
    plt.plot(wl_clean*1e7, ratio_smooth, label="Smoothed BS Ratio")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative beam splitter ratio (P1/P2)")
    plt.title("Beam Splitter Spectral Asymmetry (Filtered)")
    plt.legend()
    plt.show()


def main():
    wl_env, int_env = read_envelope_data()
    wl_gr, int_gr = get_grating_data()

    # sort arrays
    env_sort = np.argsort(wl_env)
    wl_env = wl_env[env_sort]
    int_env = int_env[env_sort]

    gr_sort = np.argsort(wl_gr)
    wl_gr = wl_gr[gr_sort]
    int_gr = int_gr[gr_sort]

    # interpolate grating spectrum onto envelope grid
    interp_func = interp1d(
        wl_gr,
        int_gr,
        bounds_error=False,
        fill_value=np.nan
    )

    int_gr_interp = interp_func(wl_env)

    mask = ~np.isnan(int_gr_interp)

    wl_common = wl_env[mask]
    env_common = int_env[mask]
    gr_common = int_gr_interp[mask]

    # avoid noise division
    threshold = 0.05 * np.max(gr_common)
    valid = gr_common > threshold

    wl_valid = wl_common[valid]
    env_valid = env_common[valid]
    gr_valid = gr_common[valid]

    # compute detector response
    ratio = env_valid / gr_valid

    # smooth detector response
    ratio_smooth = savgol_filter(ratio, 21, 3)

    # normalize ratio
    ratio_smooth /= np.max(ratio_smooth)

    # reconstruct envelope
    reconstructed_raw = gr_valid * ratio_smooth

    # relative scaling factor
    scale_factor = np.mean(env_valid) / np.mean(reconstructed_raw)

    reconstructed = reconstructed_raw * scale_factor

    # ---------------------------
    # Plot 1 : detector ratio
    # ---------------------------

    plt.figure(figsize=(8,5))
    plt.plot(wl_valid, ratio_smooth)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative detector response")
    plt.title("Estimated detector transmission")
    plt.show()

    # ---------------------------
    # Plot 2 : envelope vs reconstruction
    # ---------------------------

    plt.figure(figsize=(8,5))
    plt.plot(wl_valid, env_valid, label="Envelope (interferometer)")
    plt.plot(wl_valid, reconstructed, '--', label="Grating × response (scaled)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.title("Envelope vs reconstructed spectrum")
    plt.xlim(430, 660)
    plt.show()

    # ---------------------------
    # Plot 3 : beam splitter ratio
    # ---------------------------

    get_beam_splitter_ratio()


if __name__ == "__main__":
    main()