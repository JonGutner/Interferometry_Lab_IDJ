import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sps
import scipy.interpolate as spi

import read_data_results3 as rd
import Read_spectrum as rdsp


# -----------------------------
# Helpers: FT pipeline
# -----------------------------

def center_zpd(x, y):
    """
    Shift interferogram so the centerburst (max |y|) is at the middle index.
    Also shift x so that ZPD corresponds to x=0.
    """
    i0 = int(np.argmax(np.abs(y)))
    shift = (len(y) // 2) - i0
    y2 = np.roll(y, shift)
    x2 = x - x[i0]
    return x2, y2


def ft_spectrum_from_interferogram(
    x_opd_m,
    y,
    N=2**19,
    window="blackmanharris",
    detrend_type="linear",
    interp_kind="linear",
    use_real_spectrum=True,
    clip_negative=True,
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


def data_i(
    file,
    metres_per_microstep=3.66e-11,
    N=2**19,
    window="hann",
    use_real_spectrum=True,
):

    results = rd.read_data3("data/" + file + ".txt")

    y1 = np.array(results[1], dtype=float)
    x_microsteps = np.array(results[5], dtype=float)
    x_opd_m = x_microsteps * metres_per_microstep

    nu, intensity, meta = ft_spectrum_from_interferogram(
        x_opd_m,
        y1,
        N=N,
        window=window,
        detrend_type="linear",
        interp_kind="linear",
        use_real_spectrum=use_real_spectrum,
        clip_negative=True,
    )

    return [file, nu, intensity, meta]


def data_g(file):
    """
    Load already-computed grating spectrum.
    Handles both comma and semicolon delimiters automatically.
    Returns: [name, wavelength_array(m), intensity_array]
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

    # ---- FIX: convert nm → meters ----
    wl = np.array(wavelengths) * 1e-9
    intensity = np.array(intensities)

    return [file, wl, intensity]


def to_wavelength(x_nu):

    x_nu = np.asarray(x_nu, dtype=float)

    m = x_nu > 0
    wl = np.empty_like(x_nu)
    wl[:] = np.nan

    wl[m] = 1.0 / x_nu[m]

    return wl


def extract_envelope(y, polyorder=3):

    n = len(y)

    window = max(polyorder + 2, (n // 3) | 1)

    if window % 2 == 0:
        window += 1

    window = min(window, n - (1 - n % 2))

    if window % 2 == 0:
        window -= 1

    window = max(window, polyorder + 2 + (polyorder + 2) % 2)

    print(f"  Envelope window: {window} samples over {n} points")

    return sps.savgol_filter(y, window_length=window, polyorder=polyorder)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    files_i_input = input("Interferogram files - need FT (comma separated, blank to skip): ").strip()
    files_g_input = input("Grating files - already spectra (comma separated, blank to skip): ").strip()

    files_i = [f.strip() for f in files_i_input.split(",") if f.strip()] if files_i_input else []
    files_g = [f.strip() for f in files_g_input.split(",") if f.strip()] if files_g_input else []

    metres_per_microstep = 3.66e-11
    N = 2**19
    window = "hann"
    use_real_spectrum = True

    if files_i and files_g:

        name_nom, nu_nom, intensity_nom, _ = data_i(
            files_i[0],
            metres_per_microstep=metres_per_microstep,
            N=N,
            window=window,
            use_real_spectrum=use_real_spectrum,
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


    interferograms = []

    for file in files_i:

        name, nu, intensity, meta = data_i(
            file,
            metres_per_microstep=metres_per_microstep_fitted,
            N=N,
            window=window,
            use_real_spectrum=use_real_spectrum,
        )

        wl = to_wavelength(nu)

        interferograms.append((name, wl, intensity))

        print(f"\n--- {name} (interferogram) ---")
        print("Diagnostics:", meta)


    gratings = []

    for file in files_g:

        name, wl, intensity = data_g(file)

        gratings.append((name, wl, intensity))

    pairs = []

    for i_name, i_wl, i_intensity in interferograms:

        i_lower = i_name.lower()

        for g_name, g_wl, g_intensity in gratings:

            g_lower = g_name.lower()

            # Match colours only
            if ("white" in i_lower and "white" in g_lower) or \
                    ("yellow" in i_lower and "yellow" in g_lower) or \
                    ("green" in i_lower and "green" in g_lower):
                vis_mask = np.isfinite(i_wl) & (i_wl > 380e-9) & (i_wl < 750e-9)

                wl_vis = i_wl[vis_mask]
                int_vis = i_intensity[vis_mask]

                g_vis = np.interp(wl_vis, g_wl, g_intensity, left=np.nan, right=np.nan)
                g_vis_clean = np.where(np.isfinite(g_vis), g_vis, 0.0)

                print(f"\nRatio {i_name} / {g_name}:")

                blue_env_full = extract_envelope(int_vis)
                orange_env_full = extract_envelope(g_vis_clean)

                blue_env_full = np.clip(blue_env_full, 0, None)
                orange_env_full = np.clip(orange_env_full, 1e-10, None)

                ratio_full = blue_env_full / orange_env_full

                ratio_mask = (wl_vis > 430e-9) & (wl_vis < 600e-9) & np.isfinite(g_vis)

                wl_ratio = wl_vis[ratio_mask]
                ratio_reliable = ratio_full[ratio_mask]

                ratio_plot = extract_envelope(ratio_reliable)

                ratio_on_i = np.interp(
                    i_wl,
                    wl_ratio,
                    ratio_reliable,
                    left=ratio_reliable[0],
                    right=ratio_reliable[-1],
                )

                ratio_on_i = np.clip(ratio_on_i, 1e-10, None)

                corrected_i = i_intensity / ratio_on_i

                pairs.append(
                    (
                        i_name,
                        i_wl,
                        i_intensity,
                        corrected_i,
                        g_name,
                        g_wl,
                        g_intensity,
                        wl_ratio,
                        ratio_reliable,
                        ratio_plot,
                    )
                )

    plt.figure("Spectra")
    plt.title("Corrected interferometer vs grating")

    for (i_name, i_wl, i_intensity, corrected_i,
         g_name, g_wl, g_intensity,
         wl_ratio, ratio_reliable, ratio_plot) in pairs:

        m = np.isfinite(i_wl)

        plt.plot(i_wl[m], corrected_i[m], label=f"{i_name} corrected")
        plt.plot(g_wl, g_intensity, linestyle='--', label=g_name)


    paired_g_names = [p[4] for p in pairs]

    for g_name, g_wl, g_intensity in gratings:

        if g_name not in paired_g_names:

            plt.plot(g_wl, g_intensity, linestyle='--', label=g_name)


    plt.xlim(3.5e-7, 8e-7)
    plt.xlabel("Wavelength (m)")
    plt.ylabel("Intensity (a.u.)")
    plt.grid(True)
    plt.legend()


    if pairs:

        plt.figure("Smoothed envelope ratio (430-600nm)")
        plt.title("Smoothed envelope ratio: interferogram / grating (430-600nm)")

        for (i_name, i_wl, i_intensity, corrected_i,
             g_name, g_wl, g_intensity,
             wl_ratio, ratio_reliable, ratio_plot) in pairs:

            plt.plot(wl_ratio, ratio_reliable, alpha=0.3, label=f"{i_name} / {g_name} (raw)")
            plt.plot(wl_ratio, ratio_plot, linewidth=2, label=f"{i_name} / {g_name} (smoothed)")


        plt.xlabel("Wavelength (m)")
        plt.ylabel("Ratio (a.u.)")
        plt.grid(True)
        plt.legend()


    plt.show()