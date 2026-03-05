import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import Read_spectrum as rdsp


def data(file, metres_per_microstep=3.67e-11):
    # Step 1 get the data and the x position
    results = rd.read_data3('data/' + file + '.txt')

    # Describe the global calibration used (from either Task 6, or crossing_points.py)
    # metres_per_microstep = 2.0*metres_per_microstep  # if from Task 6

    # We are only going to need data from one detector
    # make sure the index is the right one for your data!
    y1 = np.array(results[1])

    # get x-axis data from the results
    x = np.array(results[5]) * metres_per_microstep

    # centre the y-axis on zero by either subtracting the mean
    y1 = y1 - y1.mean()

    # Butterworth filter to correct for offset
    # filter_order = 2
    # freq = 1 #cutoff frequency
    # sampling = 50 # sampling frequency
    # sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
    # filtered = signal.sosfilt(sos, y1)
    # y1 = filtered

    # Hann Window to reduce artifacts
    y2 = y1 * np.hanning(len(y1))

    # Cubic Spline part - the FFT requires a regular grid on the x-axis
    N = int(1e7)
    xs = np.linspace(x[0], x[-1], N)
    y = y2[:len(x)]
    cs = spi.CubicSpline(x, y)

    # FFT to extract spectra
    yf1 = spf.fft(cs(xs))
    xf1 = spf.fftfreq(len(xs))
    xf1 = spf.fftshift(xf1)
    yf1 = spf.fftshift(yf1)
    xx1 = xf1[int(len(xf1) / 2 + 1):len(xf1)]

    # convert x-axis to wavelength
    distance = xs[1:] - xs[:-1]
    repx1 = distance.mean() / xx1

    # Zero out noise above 1000nm
    yf1_out = abs(yf1[int(len(xf1) / 2 + 1):len(xf1)])
    yf1_out[abs(repx1) > 1000e-9] = 0

    return [file, abs(repx1), yf1_out]


def grating(file):
    results = rdsp.read_data4('data/' + file + '.txt')
    return [file, results[0], results[1]]


files_i = str(input()).split(",")  # data from interferogram
files_g = str(input()).split(",")  # data from grating
title = 'Data from: '


def plots(files, title, x, metres_per_microstep=3.67e-11):
    for file in files:
        if x == 'i':
            item = data(file, metres_per_microstep)
        if x == 'g':
            item = grating(file)
        title += item[0]
        if file != files[:-1]:
            title += ' and '
        print(item[1][np.argmax(item[2])])
        return [item[1], np.array(item[2]), item[0]]  # raw amplitudes, no normalisation


def plots_norm(files, title, x):
    for file in files:
        if x == 'i':
            item = data(file)
        if x == 'g':
            item = grating(file)
        title += item[0]
        if file != files[:-1]:
            title += ' and '
        plt.plot(item[1], np.log(item[2]), label=item[0])
        print(item[1][np.argmax(item[2])])


def atten(files, title):
    item_1 = data(files[0])
    title += item_1[0] + ' and '
    plt.plot(item_1[1], item_1[2], label=item_1[0])
    yellow_y_max = item_1[2].max()
    yellow_x_max = item_1[1][np.argmax(item_1[2])]

    item_2 = data(files[1])
    title += item_2[0]
    plt.plot(item_2[1], item_2[2], label=item_2[0])
    x2 = item_2[1]
    y2 = item_2[2]

    if x2[0] > x2[-1]:
        x2 = x2[::-1]
        y2 = y2[::-1]

    white_equiv = np.interp(yellow_x_max, x2, y2)
    print(white_equiv, yellow_y_max, yellow_x_max)
    atten_ratio = white_equiv / yellow_y_max
    print(atten_ratio)
    print(np.sum(item_2[2]) / np.sum(item_1[2]))


# ---------------------------------------------------------------
# SINGLE-POINT CALIBRATION USING 450nm PEAK
# ---------------------------------------------------------------
iplot_nominal = plots(files_i, title, 'i', 3.67e-11)
gplot = plots(files_g, title, 'g')

# Find blue peak in 420-480nm window
mask_peak = (iplot_nominal[0] > 420e-9) & (iplot_nominal[0] < 480e-9)
blue_peak_wl = iplot_nominal[0][mask_peak][np.argmax(iplot_nominal[1][mask_peak])]

# Find orange peak in same window
orange_wl = np.array(gplot[0])
orange_amp = np.array(gplot[1])
mask_peak_g = (orange_wl > 420e-9) & (orange_wl < 480e-9)
orange_peak_wl = orange_wl[mask_peak_g][np.argmax(orange_amp[mask_peak_g])]

print(f"Blue peak wavelength (nominal):  {blue_peak_wl*1e9:.2f} nm")
print(f"Orange peak wavelength:          {orange_peak_wl*1e9:.2f} nm")

scale_factor = orange_peak_wl / blue_peak_wl
metres_per_microstep_fitted = 3.67e-11 * scale_factor
print(f"Scale factor:                    {scale_factor:.6f}")
print(f"Fitted metres_per_microstep:     {metres_per_microstep_fitted:.4e}")

# Second pass with corrected calibration
iplot = plots(files_i, title, 'i', metres_per_microstep_fitted)

# Interpolate grating onto interferometer wavelength axis
new_g = np.interp(iplot[0], gplot[0], gplot[1])

# ---------------------------------------------------------------
# FFT NOTCH FILTER ON NORMALISED RATIO IN WAVENUMBER SPACE
# ---------------------------------------------------------------

# 1. Restrict to grating-reliable range
mask_reliable = (iplot[0] > 430e-9) & (iplot[0] < 600e-9)
print(f"Points in reliable window (430-600nm): {np.sum(mask_reliable)}")
wl_reliable = iplot[0][mask_reliable]
blue_reliable = iplot[1][mask_reliable]
orange_reliable = new_g[mask_reliable]

# 2. Compute normalised ratio
ratio_reliable = (blue_reliable / blue_reliable.max()) / (orange_reliable / orange_reliable.max())

# 3. Resample onto uniform wavenumber grid
sigma_reliable = 1.0 / wl_reliable
sort_idx = np.argsort(sigma_reliable)
sigma_sorted = sigma_reliable[sort_idx]
ratio_sorted = ratio_reliable[sort_idx]
sigma_uniform = np.linspace(sigma_sorted[0], sigma_sorted[-1], len(sigma_sorted))
ratio_uniform = np.interp(sigma_uniform, sigma_sorted, ratio_sorted)

# 4. FFT in wavenumber space
n = len(ratio_uniform)
d_sigma = sigma_uniform[1] - sigma_uniform[0]
ratio_fft = np.fft.fft(ratio_uniform)
freqs_sigma = np.fft.fftfreq(n, d=d_sigma)  # units: metres (OPD)

# Search only positive frequencies
pos_magnitude = np.abs(ratio_fft[1:n//2])
pos_freqs = freqs_sigma[1:n//2]
fringe_idx_pos = np.argmax(pos_magnitude)
fringe_idx_full = fringe_idx_pos + 1
fringe_freq = freqs_sigma[fringe_idx_full]
print(f"Fringe frequency: {fringe_freq*1e6:.4f} µm")

# ---------------------------------------------------------------
# ADJUSTABLE NOTCH WIDTH - tune this parameter
notch_half_width_um = 1   # half-width in µm either side of fringe peak
                              # start small (0.1-0.5 µm), increase if fringes persist
# ---------------------------------------------------------------

# Convert notch half-width from µm to OPD frequency units (metres)
notch_half_width_m = notch_half_width_um * 1e-6

# Find all positive frequency indices within notch window
notch_mask_pos = np.abs(pos_freqs - fringe_freq) <= notch_half_width_m
notch_indices = np.where(notch_mask_pos)[0] + 1  # +1 to map back to full array

print(f"Notch half-width: {notch_half_width_um} µm")
print(f"Notch covers {fringe_freq*1e6 - notch_half_width_um:.4f} to "
      f"{fringe_freq*1e6 + notch_half_width_um:.4f} µm "
      f"({len(notch_indices)} frequency bins)")

# 5. Apply notch - zero spike and conjugate mirror
ratio_fft_filtered = ratio_fft.copy()
for idx in notch_indices:
    ratio_fft_filtered[idx] = 0          # positive frequency
    ratio_fft_filtered[n - idx] = 0      # conjugate mirror

# Add to the notch application loop
second_notch_centre_um = 5.0
second_notch_width_um = 3.0  # broad to cover the hump
second_notch_mask = np.abs(pos_freqs - second_notch_centre_um * 1e-6) <= second_notch_width_um * 1e-6
second_notch_indices = np.where(second_notch_mask)[0] + 1
for idx in second_notch_indices:
    ratio_fft_filtered[idx] = 0
    ratio_fft_filtered[n - idx] = 0

# DIAGNOSTIC: FFT before and after notch
plt.figure('Ratio FFT in wavenumber space')
plt.plot(pos_freqs * 1e6, pos_magnitude, label='original', alpha=0.6)
plt.plot(pos_freqs * 1e6, np.abs(ratio_fft_filtered[1:n//2]), label='filtered', linewidth=2)
plt.axvline(fringe_freq * 1e6, color='r', linestyle='--',
            label=f'Fringe at {fringe_freq*1e6:.3f} µm')
plt.axvspan((fringe_freq - notch_half_width_m) * 1e6,
            (fringe_freq + notch_half_width_m) * 1e6,
            alpha=0.15, color='red', label=f'Notch ±{notch_half_width_um} µm')
plt.xlabel('OPD (µm)')
plt.ylabel('Magnitude')
plt.title('FFT of normalised ratio in wavenumber space')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()

# 6. Inverse FFT
ratio_clean_sigma = np.real(np.fft.ifft(ratio_fft_filtered))

# 7. Resample back onto original wavelength axis
ratio_clean_wl = np.interp(sigma_reliable, sigma_uniform, ratio_clean_sigma)

# DIAGNOSTIC: ratio before and after notch
plt.figure('Normalised ratio diagnostic')
plt.plot(wl_reliable, ratio_reliable / ratio_reliable.max(), label='ratio (original)', alpha=0.6)
plt.plot(wl_reliable, ratio_clean_wl / ratio_clean_wl.max(), label='ratio (fringe removed)', linewidth=2)
plt.xlabel('Wavelength (m)')
plt.ylabel('Normalised ratio')
plt.title('Normalised ratio diagnostic (430-600nm)')
plt.legend()
plt.grid()
plt.show()

# 8. Apply correction to full blue spectrum
ratio_full = np.interp(iplot[0], wl_reliable[::-1], ratio_clean_wl[::-1],
                       left=ratio_clean_wl[-1], right=ratio_clean_wl[0])
ratio_full = np.where(np.abs(ratio_full) > 1e-6 * np.abs(ratio_full).max(),
                      ratio_full, 1e-6 * np.abs(ratio_full).max())
blue_corrected = iplot[1] / ratio_full
blue_corrected_norm = blue_corrected / blue_corrected.max()

# ---------------------------------------------------------------

# plots_norm(files_i, title, 'i')
# plots_norm(files_i, title, 'g')
# atten(file_is, title)

plt.figure('Spectrum using global calibration FFT')
plt.plot(iplot[0], iplot[1] / iplot[1].max(), label=iplot[2] + ' original', alpha=0.4)
plt.plot(gplot[0], np.array(gplot[1]) / np.array(gplot[1]).max(), label=gplot[2])
plt.plot(iplot[0], blue_corrected_norm, label='corrected', linewidth=2)
plt.title(title)
plt.xlim(3.5e-7, 8e-7)
plt.xlabel('Wavelength (m)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.grid()
plt.savefig('figures/temp_data.png')
plt.show()