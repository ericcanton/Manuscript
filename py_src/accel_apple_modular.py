import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import hilbert, find_peaks, spectrogram
from scipy.ndimage import uniform_filter1d

def preprocess_accel(accel_df):
    # Ensure time is in seconds (int or float)
    if np.issubdtype(accel_df["time"].dtype, np.datetime64):
        accel_df = accel_df.copy()
        accel_df["time"] = pd.to_datetime(accel_df["time"]).astype(np.int64) // 10**9

    x = np.nan_to_num(accel_df["x"].values)
    y = np.nan_to_num(accel_df["y"].values)
    z = np.nan_to_num(accel_df["z"].values)
    t = accel_df["time"].values
    return t, x, y, z

def compute_motion_bpm(t, x, y, z, time_resolution=30):
    fs = len(t) / (t[-1] - t[0])
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_filt = np.diff(mag, prepend=mag[0])
    mag_filt = np.sqrt(uniform_filter1d(mag_filt**2, size=18))
    analytic_signal = hilbert(mag_filt)
    phase = np.angle(analytic_signal)
    window_size = max(int(0.8 * fs), 1)
    phase_mean = uniform_filter1d(phase, size=window_size)
    phase = phase - phase_mean

    f, t_spec, Sxx = spectrogram(
        phase,
        fs=fs,
        nperseg=int(time_resolution * fs),
        noverlap=int(0.5 * time_resolution * fs),
        scaling="density",
        mode="magnitude",
    )
    freq_mask = (f >= 0.5) & (f <= 3.5)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    p = Sxx / np.maximum(Sxx.max(axis=0, keepdims=True), 1e-12)

    idx = np.zeros(len(t_spec), dtype=int)
    width = np.zeros(len(t_spec))
    for i in range(len(t_spec)):
        peaks, properties = find_peaks(p[:, i], height=0.9)
        if len(peaks) == 0:
            idx[i] = 300
            width[i] = np.nan
        else:
            idx[i] = peaks[-1]
            width[i] = properties["widths"][-1] if "widths" in properties and len(properties["widths"]) > 0 else np.nan

    HR = 60 * f[idx]
    for i in range(1, len(HR)):
        if np.abs(HR[i] - HR[i - 1]) > 30:
            HR[i] = HR[i - 1]
    HR_smooth = uniform_filter1d(HR, size=5)
    return f, t_spec, p, HR_smooth, width, fs

def interpolate_hr_to_spec(hr_df, t_spec):
    # Interpolate HR BPM to t_spec (spectrogram time axis)
    hr_times = hr_df["time"].values
    hr_bpm = hr_df["bpm"].values
    interp_bpm = np.interp(t_spec, hr_times, hr_bpm)
    return interp_bpm

def plot_results(f, t_spec, p, HR_smooth, hr_df=None, fs=None,
                 save_to=None):
    plt.figure(figsize=(10, 6))
    extent = [t_spec[0], t_spec[-1], f[0] * 60, f[-1] * 60]
    plt.imshow(
        p,
        aspect="auto",
        origin="lower",
        extent=extent,
        alpha=0.4,
        cmap="viridis"
    )
    plt.scatter(t_spec, HR_smooth, s=5, color="red", label="Motion BPM")
    plt.xlabel("Time (s)")
    plt.ylabel("Motion BPM/RPM")
    plt.title("Motion BPM/RPM over Time")

    if hr_df is not None:
        interp_bpm = interpolate_hr_to_spec(hr_df, t_spec)
        plt.plot(t_spec, interp_bpm, color="white", lw=1.5, label="Ground Truth HR")
        # Optionally, overlay pure sine/cosine at HR frequency (as a visual guide)
        # freq = interp_bpm / 60
        # for i in range(len(t_spec)):
        #     plt.axhline(interp_bpm[i], color="white", alpha=0.1)

    plt.legend()
    plt.colorbar(label="Normalized Power Spectral Density")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()

def analyze_accel_vs_hr(accel_df, hr_df=None, time_resolution=30,
                        save_to=None):
    t, x, y, z = preprocess_accel(accel_df)
    f, t_spec, p, HR_smooth, width, fs = compute_motion_bpm(t, x, y, z, time_resolution)
    if hr_df is not None:
        # Ensure hr_df columns are named "time" and "bpm"
        if "bpm" not in hr_df.columns:
            hr_df = hr_df.rename(columns={hr_df.columns[1]: "bpm"})
    plot_results(f, t_spec, p, HR_smooth, hr_df=hr_df, fs=fs,
                 save_to=save_to)
    print("std_width_apple:", np.nanstd(width))
    return {
        "f": f,
        "t_spec": t_spec,
        "p": p,
        "HR_smooth": HR_smooth,
        "width": width,
        "fs": fs,
    }

if __name__ == "__main__":
    from pathlib import Path
    from pisces.data_sets import DataSetObject
    # Example usage:
    # Load your accelerometer and heart rate data into pandas DataFrames
    # data_dir = Path("your_data_directory")  # replace with your data directory
    data_dir = Path("/Users/eric/Engineering/Work/pisces/data/walch_et_al")  # replace with your data directory

    data_set_name = "walch_et_al"
    
    cwd = Path.cwd()
    
    datasets = DataSetObject.find_data_sets(data_dir)
    this_set = datasets.get(data_set_name)
    if this_set is None:
        raise ValueError(f"Dataset {data_set_name} not found in {data_dir}")
    this_set.parse_data()
    print(f"Loaded dataset from {data_dir}")
    recording_pairs = []
    subject_ids = []

    ids_to_use = this_set.ids
    # ids_to_use = ['1455390']
    
    for idno in ids_to_use:
        try:
            accel_data = this_set.get_feature_data('accelerometer', idno)
            psg_data = this_set.get_feature_data('psg', idno)
            hr_data = this_set.get_feature_data('heartrate', idno)
            if accel_data is None or psg_data is None:
                print(f"Missing data for ID {idno}. Skipping...")
                continue
            print(f"ID: {idno}, Accel samples: {len(accel_data)}, PSG samples: {len(psg_data)}")

            accel_df = accel_data.to_pandas()
            hr_df = hr_data.to_pandas() if hr_data is not None else None
            
            # set columns to time, x, y, z
            time_col = "time"
            accel_df.columns = [time_col, "x", "y", "z"]
            if hr_df is not None:
                hr_df.columns = [time_col, "bpm"]
            analyze_accel_vs_hr(accel_df, hr_df,
                                time_resolution=30,
                                save_to=cwd / f"{idno}_accel_vs_hr.png")
        except Exception as e:
            print(f"Error processing ID {idno}: {e}")
            continue
# Example usage:
# accel_df = pd.read_csv("your_accel.csv")  # columns: time, x, y, z
# hr_df = pd.read_csv("your_hr.csv")        # columns: time, bpm
# analyze_accel_vs_hr(accel_df, hr_df)