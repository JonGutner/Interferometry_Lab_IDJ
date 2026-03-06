import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sps
import scipy.interpolate as spi
import scipy.fftpack as spf

import read_data_results3 as rd


# -----------------------------
# Helpers: FT pipeline
# -----------------------------

def center_zpd(x, y):
    i0 = int(np.argmax(np.abs(y)))
    shift = (len(y) // 2) - i0
    y2 = np.roll(y, shift)
    x2 = x - x[i0]
    return x2, y2


def ft_spectrum_from_interferogram(
    x_opd_m, y, N=2**19, window="hann",
    detrend_type="linear", interp_kind="linear",
    use_real_spectrum=True, clip_negative=True,
):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_opd_m, dtype=float)
    y = sps.detrend(y, type=detrend_type)
    x, y = center_zpd(x, y)

    if window == "hann":
        w = np.hanning(len(y))
    elif window == "blackmanharris":
        w = sps.windows.blackmanharris(len(y))
    elif window == "kaiser14":
        w = sps.windows.kaiser(len(y), beta=14.0)
    else:
        raise ValueError("window must be one of: 'hann', 'blackmanharris', 'kaiser14'")
    y = y * w

    xs = np.linspace(x.min(), x.max(), int(N))
    f = spi.interp1d(x, y, kind=interp_kind, fill_value="extrapolate")
    ys = f(xs)
    dx = xs[1] - xs[0]

    Y = np.fft.rfft(ys)
    nu = np.fft.rfftfreq(len(xs), d=dx)
    mask = nu > 0
    nu = nu[mask]
    Y = Y[mask]

    if use_real_spectrum:
        intensity = np.real(Y)
        if clip_negative:
            intensity = intensity.copy()
            intensity[intensity < 0] = 0.0
    else:
        intensity = np.abs(Y)

    OPDmax = float(np.max(np.abs(xs)))
    expected_dnu = (1.0 / (2.0 * OPDmax)) if OPDmax > 0 else np.nan
    meta = {
        "dx_m": float(dx),
        "OPDmax_m": OPDmax,
        "expected_dnu_cycles_per_m": float(expected_dnu),
        "N_uniform": int(len(xs)),
    }
    return nu, intensity, meta


def data_i(file, metres_per_microstep=3.66e-11, N=2**19, window="hann", use_real_spectrum=True):
    results = rd.read_data3("data/" + file + ".txt")
    y1 = np.array(results[1], dtype=float)
    x_microsteps = np.array(results[5], dtype=float)
    x_opd_m = x_microsteps * metres_per_microstep
    nu, intensity, meta = ft_spectrum_from_interferogram(
        x_opd_m, y1, N=N, window=window,
        detrend_type="linear", interp_kind="linear",
        use_real_spectrum=use_real_spectrum, clip_negative=True,
    )
    return [file, nu, intensity, meta]


def data_g(file):
    """
    Load grating spectrum. Handles comma and semicolon delimiters.
    Converts wavelength from nm to metres.
    """
    path = "data/" + file + ".txt"
    wavelengths, intensities = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for delim in (',', ';'):
                parts = line.split(delim)
                if len(parts) == 2:
                    try:
                        wavelengths.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                        break
                    except ValueError:
                        continue
    wl = np.array(wavelengths) * 1e-9
    return [file, wl, np.array(intensities)]


def to_wavelength(x_nu):
    x_nu = np.asarray(x_nu, dtype=float)
    m = x_nu > 0
    wl = np.empty_like(x_nu)
    wl[:] = np.nan
    wl[m] = 1.0 / x_nu[m]
    return wl


def remove_fringes_via_ft(wl, intensity, label="",
                          notch_opd_um=None,
                          notch_half_width_um=0.5):
    """
    Remove periodic etalon fringes from a spectrum using FT in wavenumber space.

    Method:
      1. Resample spectrum onto uniform wavenumber (sigma = 1/lambda) grid.
         Fringes are periodic in sigma, not lambda, so the FT spike is sharp.
      2. Compute FT of the resampled spectrum.
      3. Plot the FT magnitude so you can see the fringe spike.
      4. If notch_opd_um is given, zero the spike at that OPD and its mirror.
         If None, skip the notch and return the original spectrum
         (diagnostic mode — run first to identify the spike location).
      5. Inverse FT to recover the cleaned spectrum.
      6. Resample back onto the original wavelength axis.

    Parameters:
      notch_opd_um       : OPD of fringe spike in microns. Set to None first
                           to identify it from the FT plot, then set it.
      notch_half_width_um: half-width of notch in microns either side of spike.
                           Start with 0.5, widen if fringes persist.
    """
    wl = np.asarray(wl, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # Only work on finite values
    m = np.isfinite(wl) & np.isfinite(intensity) & (wl > 0)
    wl_clean = wl[m]
    int_clean = intensity[m]

    # Sort by wavelength ascending
    idx = np.argsort(wl_clean)
    wl_clean = wl_clean[idx]
    int_clean = int_clean[idx]

    # 1. Convert to wavenumber and resample onto uniform grid
    # wl_clean is ascending -> sigma is descending, must flip both for interp
    sigma = 1.0 / wl_clean          # wavenumber (cycles/m), descending
    sigma_asc = sigma[::-1]         # ascending
    int_asc   = int_clean[::-1]     # matching order
    n = len(sigma_asc)
    sigma_uniform = np.linspace(sigma_asc[0], sigma_asc[-1], n)
    d_sigma = sigma_uniform[1] - sigma_uniform[0]
    int_uniform = np.interp(sigma_uniform, sigma_asc, int_asc)

    # 2. FT in wavenumber space
    INT = np.fft.rfft(int_uniform)
    freqs = np.fft.rfftfreq(n, d=d_sigma)  # units: metres (OPD)
    magnitude = np.abs(INT)

    print(f"\n  [{label}] FT in wavenumber space:")
    print(f"    Points: {n},  d_sigma: {d_sigma:.3e} m^-1")
    print(f"    OPD resolution: {1.0/(n*d_sigma)*1e6:.4f} µm")
    print(f"    Max OPD visible: {freqs[-1]*1e6:.1f} µm")

    # 3. Diagnostic: plot FT magnitude
    # Screen positive frequencies only (skip DC at index 0)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    magnitude_pos = magnitude[pos_mask]

    plt.figure(f"FT of spectrum in wavenumber space - {label}")
    plt.title(f"FT of spectrum in wavenumber space\n{label}  "
              f"(identify fringe spike OPD, then set notch_opd_um)")
    plt.plot(freqs_pos * 1e6, magnitude_pos)
    if notch_opd_um is not None:
        opd_list = [notch_opd_um] if np.isscalar(notch_opd_um) else notch_opd_um
        hw_list  = [notch_half_width_um] if np.isscalar(notch_half_width_um) else notch_half_width_um
        if np.isscalar(notch_half_width_um):
            hw_list = [notch_half_width_um] * len(opd_list)
        for i, (opd, hw) in enumerate(zip(opd_list, hw_list)):
            plt.axvspan(
                opd - hw, opd + hw,
                alpha=0.2, color='red',
                label=f"Notch +/-{hw} um at {opd} um" if i == 0 else f"Notch +/-{hw} um at {opd} um"
            )
        plt.legend()
    plt.xscale('log')
    plt.xlabel("OPD (um)  [log scale]")
    plt.ylabel("Magnitude")
    plt.grid(True, which='both')

    # 4. If no notch specified, return original spectrum (diagnostic mode)
    if notch_opd_um is None:
        print("    notch_opd_um not set - diagnostic mode, no cleaning applied")
        return wl, intensity

    # Accept scalar or list for notch_opd_um and notch_half_width_um
    if np.isscalar(notch_opd_um):
        notch_opd_um = [notch_opd_um]
    if np.isscalar(notch_half_width_um):
        notch_half_width_um = [notch_half_width_um] * len(notch_opd_um)

    INT_filtered = INT.copy()
    for opd, hw in zip(notch_opd_um, notch_half_width_um):
        notch_opd_m = opd * 1e-6
        notch_hw_m  = hw  * 1e-6
        notch_mask  = np.abs(freqs - notch_opd_m) <= notch_hw_m
        n_notched   = np.sum(notch_mask)
        INT_filtered[notch_mask] = 0
        print(f"    Notch at {opd} um +/-{hw} um: {n_notched} bins zeroed")

    # 5. Inverse FT
    int_clean_sigma = np.real(np.fft.irfft(INT_filtered, n=n))

    # 6. Resample back onto original wavelength axis
    # sigma_uniform and int_clean_sigma are in ascending sigma order
    # sigma_asc is also ascending — interp directly, then flip back to match wl_clean order
    int_clean_wl_asc = np.interp(sigma_asc, sigma_uniform, int_clean_sigma)
    int_clean_wl = int_clean_wl_asc[::-1]  # back to descending sigma (= ascending wl)

    # Map back onto original (possibly non-uniform) wavelength array
    result = intensity.copy()
    result[m] = int_clean_wl[np.argsort(idx)]   # undo the sort
    result = np.clip(result, 0, None)

    return wl, result


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    files_i_input = input("Interferogram files - need FT (comma separated, blank to skip): ").strip()
    files_g_input = input("Grating files - already spectra (comma separated, blank to skip): ").strip()

    files_i = [f.strip() for f in files_i_input.split(",") if f.strip()] if files_i_input else []
    files_g = [f.strip() for f in files_g_input.split(",") if f.strip()] if files_g_input else []

    # ---------------------------------------------------------------
    # SETTINGS
    # ---------------------------------------------------------------
    metres_per_microstep = 3.66e-11
    N = 2**19
    window = "hann"
    use_real_spectrum = True

    # Fringe removal settings — run once with notch_opd_um=None to identify
    # spikes in the FT plot, then set values and rerun.
    # Accepts scalar (single notch) or list (multiple notches), e.g.:
    #   notch_opd_um        = [38.0, 2.0]
    #   notch_half_width_um = [2.0,  1.0]
    notch_opd_um = None# [38.0, 1.5, 3.0, 5.0, 10.0, 20.0]
    notch_half_width_um = [2.0, 0.3, 0.3, 0.3, 0.5, 1.0]

    # ---------------------------------------------------------------
    # SINGLE-POINT CALIBRATION USING 450nm PEAK
    # ---------------------------------------------------------------
    if files_i and files_g:
        name_nom, nu_nom, intensity_nom, _ = data_i(
            files_i[0], metres_per_microstep=metres_per_microstep,
            N=N, window=window, use_real_spectrum=use_real_spectrum,
        )
        wl_nom = to_wavelength(nu_nom)
        name_g0, wl_g0, int_g0 = data_g(files_g[0])

        mask_blue = np.isfinite(wl_nom) & (wl_nom > 420e-9) & (wl_nom < 480e-9)
        blue_peak_wl = wl_nom[mask_blue][np.argmax(intensity_nom[mask_blue])]

        mask_orange = (wl_g0 > 420e-9) & (wl_g0 < 480e-9)
        orange_peak_wl = wl_g0[mask_orange][np.argmax(int_g0[mask_orange])]

        scale_factor = orange_peak_wl / blue_peak_wl
        metres_per_microstep_fitted = metres_per_microstep * scale_factor

        print("\nCalibration:")
        print(f"  Blue peak (nominal):  {blue_peak_wl*1e9:.2f} nm")
        print(f"  Orange peak:          {orange_peak_wl*1e9:.2f} nm")
        print(f"  Scale factor:         {scale_factor:.6f}")
        print(f"  Fitted m/microstep:   {metres_per_microstep_fitted:.4e}")
    else:
        metres_per_microstep_fitted = metres_per_microstep
        print("\nCalibration skipped (need both interferogram and grating files)")

    # ---------------------------------------------------------------
    # LOAD ALL FILES WITH FITTED CALIBRATION
    # ---------------------------------------------------------------
    interferograms = []
    for file in files_i:
        name, nu, intensity, meta = data_i(
            file, metres_per_microstep=metres_per_microstep_fitted,
            N=N, window=window, use_real_spectrum=use_real_spectrum,
        )
        wl = to_wavelength(nu)
        interferograms.append((name, wl, intensity))
        print(f"\n--- {name} (interferogram) ---")
        print("Diagnostics:", meta)

    gratings = []
    for file in files_g:
        name, wl, intensity = data_g(file)
        gratings.append((name, wl, intensity))

    # ---------------------------------------------------------------
    # FRINGE REMOVAL VIA FT IN WAVENUMBER SPACE
    # Applied to interferometer spectra only.
    # First run: notch_opd_um=None — inspect FT plot to find spike OPD.
    # Second run: set notch_opd_um to the spike location in µm.
    # ---------------------------------------------------------------
    interferograms_cleaned = []
    for name, wl, intensity in interferograms:
        wl_out, int_out = remove_fringes_via_ft(
            wl, intensity,
            label=name,
            notch_opd_um=notch_opd_um,
            notch_half_width_um=notch_half_width_um,
        )
        interferograms_cleaned.append((name, wl_out, int_out))

    # ---------------------------------------------------------------
    # PLOT: Raw vs cleaned interferometer spectra + grating
    # ---------------------------------------------------------------
    plt.figure("Spectra")
    plt.title("Interferometer spectra (raw vs fringe-removed) and grating")

    for (name, wl, intensity), (_, _, int_clean) in zip(interferograms, interferograms_cleaned):
        m = np.isfinite(wl)
        plt.plot(wl[m], intensity[m], alpha=0.3, label=f"{name} (raw)")
        if notch_opd_um is not None:
            plt.plot(wl[m], int_clean[m], linewidth=2, label=f"{name} (fringe removed)")

    # for g_name, g_wl, g_intensity in gratings:
    #     plt.plot(g_wl, g_intensity, linestyle='--', label=g_name)

    plt.xlim(3.5e-7, 8e-7)
    plt.xlabel("Wavelength (m)")
    plt.ylabel("Intensity (a.u.)")
    plt.grid(True)
    plt.legend()

    plt.show()