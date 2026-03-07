import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from Interferometry_Lab_IDJ import apply_global_calibration_Ivan as agci

# --- CONFIGURATION ---
# Set NORM_MODE to 'max' for Peak-Normalization or 'mean' for Mean-Normalization
NORM_MODE = 'mean'


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


def get_beam_splitter_ratio(target_wl, show_plot=True):
    bm_1 = "White_LED_Lens_SplitterRT"
    bm_2 = "White_LED_Lens_SplitterST"
    _, wl_1, int_1 = agci.data_g(bm_1)
    _, wl_2, int_2 = agci.data_g(bm_2)

    wl_1_nm, wl_2_nm = wl_1 * 1e9, wl_2 * 1e9

    interp_func = interp1d(wl_2_nm, int_2, bounds_error=False, fill_value=np.nan)
    int_2_interp = interp_func(wl_1_nm)

    ratio_raw = int_1 / int_2_interp
    ratio_smooth = savgol_filter(ratio_raw, 21, 3)

    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(wl_1_nm, ratio_smooth, label="Smoothed BS Ratio")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Relative beam splitter ratio (P1/P2)")
        plt.title("Beam Splitter Spectral Asymmetry (Diagnostic)")
        plt.ylim([.85, 1.05])
        plt.xlim(430, 660)
        plt.legend()
        plt.show()

    return interp1d(wl_1_nm, ratio_smooth, bounds_error=False, fill_value=1.0)


def main():
    wl_env, int_env = read_envelope_data()
    wl_gr, int_gr = get_grating_data()

    # Sort
    env_sort = np.argsort(wl_env)
    wl_env, int_env = wl_env[env_sort], int_env[env_sort]

    # 1. Get and apply Beam Splitter Correction
    bs_ratio_func = get_beam_splitter_ratio(wl_env, show_plot=True)
    bs_correction = bs_ratio_func(wl_env)
    int_env_corrected = int_env / bs_correction

    # Plot: BS Effect
    plt.figure(figsize=(8, 5))
    plt.plot(wl_env, int_env, label="Raw Envelope (with BS bias)")
    plt.plot(wl_env, int_env_corrected, label="Corrected Envelope (BS removed)")
    plt.title("Effect of Beam Splitter on Envelope")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.xlim(430, 660)
    plt.legend()
    plt.show()

    # 2. Interpolate grating
    interp_gr = interp1d(wl_gr, int_gr, bounds_error=False, fill_value=np.nan)
    int_gr_interp = interp_gr(wl_env)

    # 3. Masking
    mask = ~np.isnan(int_gr_interp)
    wl_common = wl_env[mask]
    env_common = int_env_corrected[mask]
    gr_common = int_gr_interp[mask]

    valid = gr_common > (0.05 * np.max(gr_common))
    wl_valid, env_valid, gr_valid = wl_common[valid], env_common[valid], gr_common[valid]

    # 4. Compute detector response
    ratio_smooth = savgol_filter(env_valid / gr_valid, 21, 3)

    # --- UPDATED NORMALIZATION LOGIC ---
    if NORM_MODE == 'max':
        detector_response = ratio_smooth / np.max(ratio_smooth)
    else:
        detector_response = ratio_smooth / np.mean(ratio_smooth)
    # ------------------------------------

    # 5. Final Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(wl_valid, detector_response, label=f"Response (Norm: {NORM_MODE})")
    plt.xlim(430, 660)
    plt.title("Corrected detector transmission")
    plt.legend()
    plt.show()

    reconstructed = (gr_valid * detector_response) * (np.mean(env_valid) / np.mean(gr_valid * detector_response))
    plt.figure(figsize=(8, 5))
    plt.plot(wl_valid, env_valid, label="Corrected Envelope")
    plt.plot(wl_valid, reconstructed, '--', label="Grating × response")
    plt.xlim(430, 660)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()