import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram

csv_path = "/Users/eric/Engineering/GitHub/Manuscript/Scripts_Supplementary/7514011923_apple_data.csv"

# Load CSV
appledata_pd = pd.read_csv(csv_path)

# Convert 'time' column to pandas datetime
appledata_pd["time"] = pd.to_datetime(appledata_pd["time"]).astype(np.int64) // 10**9

appledata = appledata_pd.to_dict(orient="list")

time_resolution = 30  # seconds

# Compute fs_apple from data
start_time = appledata["time"][0]
end_time = appledata["time"][-1]
fs_apple = len(appledata["time"]) / (end_time - start_time)

# Import data
x_apple = np.array(appledata["x"]).flatten()
y_apple = np.array(appledata["y"]).flatten()
z_apple = np.array(appledata["z"]).flatten()
index = np.arange(len(x_apple))
time_apple = index / fs_apple

# Replace nan with 0
x_apple = np.nan_to_num(x_apple)
y_apple = np.nan_to_num(y_apple)
z_apple = np.nan_to_num(z_apple)

# Compute magnitude
mag_apple = np.sqrt(x_apple**2 + y_apple**2 + z_apple**2)

# Compute derivative (jerk)
mag_filt_apple = np.diff(mag_apple, prepend=mag_apple[0])
mag_filt_apple = np.sqrt(uniform_filter1d(mag_filt_apple**2, size=18))

# Hilbert transform to get phase
analytic_signal = hilbert(mag_filt_apple)
mag_filt_apple_phase = np.angle(analytic_signal)

# Remove slow drift in phase
window_size = int(0.8 * fs_apple)
if window_size < 1:
    window_size = 1
phase_mean = uniform_filter1d(mag_filt_apple_phase, size=window_size)
mag_filt_apple_phase = mag_filt_apple_phase - phase_mean

# Spectrogram
f_apple, t_apple, Sxx = spectrogram(
    mag_filt_apple_phase,
    fs=fs_apple,
    nperseg=int(time_resolution * fs_apple),
    noverlap=int(0.5 * time_resolution * fs_apple),
    scaling="density",
    mode="magnitude",
)
# Limit frequency range to [0.5, 3.5] Hz
freq_mask = (f_apple >= 0.5) & (f_apple <= 3.5)
f_apple = f_apple[freq_mask]
Sxx = Sxx[freq_mask, :]

# Normalize each time slice
p_apple = Sxx / np.maximum(Sxx.max(axis=0, keepdims=True), 1e-12)

# Peak detection
idx = np.zeros(len(t_apple), dtype=int)
width_apple = np.zeros(len(t_apple))
for i in range(len(t_apple)):
    peaks, properties = find_peaks(p_apple[:, i], height=0.9)
    if len(peaks) == 0:
        idx[i] = 300  # fallback index
        width_apple[i] = np.nan
    else:
        idx[i] = peaks[-1]
        if "widths" in properties and len(properties["widths"]) > 0:
            width_apple[i] = properties["widths"][-1]
        else:
            width_apple[i] = np.nan

HR_apple = 60 * f_apple[idx]

# Fill outliers with previous value
for i in range(1, len(HR_apple)):
    if np.abs(HR_apple[i] - HR_apple[i - 1]) > 30:  # crude outlier threshold
        HR_apple[i] = HR_apple[i - 1]

# Moving average
window = 5
HR_apple_smooth = uniform_filter1d(HR_apple, size=window)

# Plot spectrogram
plt.figure(figsize=(10, 6))
sns.heatmap(p_apple, xticklabels=200, yticklabels=10, cmap="viridis", cbar=True)
plt.xlabel("Time (frames)")
plt.ylabel("Frequency bin")
plt.title("Spectrogram (normalized)")
plt.show()

# Overlay HR_apple
plt.figure(figsize=(10, 6))
plt.imshow(
    p_apple,
    aspect="auto",
    origin="lower",
    extent=[t_apple[0], t_apple[-1], f_apple[0] * 60, f_apple[-1] * 60],
    alpha=0.4,
)
plt.scatter(t_apple, HR_apple_smooth, s=5, color="red")
plt.xlabel("Time (s)")
plt.ylabel("Motion BPM/RPM")
plt.title("Motion BPM/RPM over Time")
plt.show()

# Scatter plot
plt.figure(figsize=(10, 4))
plt.scatter(t_apple, HR_apple_smooth, s=3)
plt.xlabel("Time (s)")
plt.ylabel("Motion BPM/RPM")
plt.title("Motion BPM/RPM")
plt.show()

# Metric calculation
std_width_apple = np.nanstd(width_apple)
print("std_width_apple:", std_width_apple)
