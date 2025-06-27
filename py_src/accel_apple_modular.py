import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import hilbert, find_peaks, spectrogram
from scipy.ndimage import uniform_filter1d
from matplotlib.widgets import SpanSelector

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
    print(f"Sampling frequency: {fs:.2f} Hz")
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_filt = np.diff(mag, prepend=mag[0])
    mag_filt = np.sqrt(uniform_filter1d(mag_filt**2, size=18))
    analytic_signal = hilbert(mag_filt)
    phase = np.angle(analytic_signal)

    # this focuses in on the heart rate frequencies
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
    # later, we will multiply by 60 to convert to BPM
    # let's work in bpm already
    # f = 60 * f  # Convert frequency to BPM
    freq_mask = (f >= 0.5) & (f <= 3.5)
    # keep 0.5 * 60, but reduce freq to 80 bpm
    # freq_mask = (f >= 30) & (f <= 200)
    # f /= 60  # Convert back to Hz for plotting
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    p = Sxx / np.maximum(Sxx.max(axis=0, keepdims=True), 1e-12)

    idx = np.zeros(len(t_spec), dtype=int)
    width = np.zeros(len(t_spec))
    for i in range(len(t_spec)):
        peaks, properties = find_peaks(p[:, i], height=0.9)
        if len(peaks) == 0:
            idx[i] = 0
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

# Breathing rate frequency range (9-30 breaths/min = 0.15-0.5 Hz)
MIN_BREATHING_FREQ = 9/60  # 9 breaths per minute
MAX_BREATHING_FREQ = 30/60  # 30 breaths per minute

def compute_motion_breathing_rate(t, x, y, z, time_resolution=60):
    fs = len(t) / (t[-1] - t[0])
    print(f"Sampling frequency: {fs:.2f} Hz")
    
    # Use raw magnitude instead of derivative for breathing
    # Breathing causes slower, larger amplitude movements
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    # Remove DC component and apply gentler filtering
    mag = np.diff(mag, prepend=mag[0])  # Use first value to avoid NaN
    mag = np.sqrt(uniform_filter1d(mag**2, size=18))  # Smoother magnitude
    mag_detrended = mag - np.mean(mag)
    mag_filt = uniform_filter1d(mag_detrended, size=int(0.5 * fs))  # Lighter smoothing
    
    # Optional: Use Hilbert transform for phase analysis
    analytic_signal = hilbert(mag_filt)
    phase = np.angle(analytic_signal)
    
    # Longer window for breathing phase detrending
    window_size = max(int(2.0 * fs), 1)  # 2-second window instead of 0.8
    phase_mean = uniform_filter1d(phase, size=window_size)
    phase = phase - phase_mean

    # Longer time windows for breathing analysis
    f, t_spec, Sxx = spectrogram(
        phase,
        fs=fs,
        nperseg=int(time_resolution * fs),  # 60-second windows
        noverlap=int(0.75 * time_resolution * fs),  # 75% overlap for breathing
        scaling="density",
        mode="magnitude",
    )
    
    freq_mask = (f >= MIN_BREATHING_FREQ) & (f <= MAX_BREATHING_FREQ)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    p = Sxx / np.maximum(Sxx.max(axis=0, keepdims=True), 1e-12)

    idx = np.zeros(len(t_spec), dtype=int)
    width = np.zeros(len(t_spec))
    for i in range(len(t_spec)):
        # Lower threshold for breathing peaks (less pronounced than HR)
        peaks, properties = find_peaks(p[:, i], height=0.7)  # Reduced from 0.9
        if len(peaks) == 0:
            idx[i] = 0
            width[i] = np.nan
        else:
            # For breathing, might want the strongest peak, not necessarily highest frequency
            peak_heights = p[peaks, i]
            strongest_peak_idx = np.argmax(peak_heights)
            idx[i] = peaks[strongest_peak_idx]
            width[i] = properties["widths"][strongest_peak_idx] if "widths" in properties and len(properties["widths"]) > 0 else np.nan

    # Convert to breaths per minute
    BR = 60 * f[idx]
    
    # More conservative outlier rejection for breathing (breathing rate changes more gradually)
    for i in range(1, len(BR)):
        if np.abs(BR[i] - BR[i - 1]) > 10:  # Reduced from 30 BPM to 10 breaths/min
            BR[i] = BR[i - 1]
    
    # Stronger smoothing for breathing rate
    BR_smooth = uniform_filter1d(BR, size=7)  # Increased from 5
    return f, t_spec, p, BR_smooth, width, fs


def compute_breathing_multiaxis(t, x, y, z, time_resolution=60):
    fs = len(t) / (t[-1] - t[0])
    
    breathing_rates = []
    
    # Analyze each axis separately
    for axis_data, axis_name in zip([x, y, z], ['x', 'y', 'z']):
        # Detrend and filter
        axis_detrended = axis_data - uniform_filter1d(axis_data, size=int(5 * fs))
        axis_filt = uniform_filter1d(axis_detrended, size=int(0.3 * fs))
        
        # Spectrogram analysis
        f, t_spec, Sxx = spectrogram(
            axis_filt,
            fs=fs,
            nperseg=int(time_resolution * fs),
            noverlap=int(0.75 * time_resolution * fs),
            scaling="density",
            mode="magnitude",
        )
        
        # Breathing frequency range
        freq_mask = (f >= 0.15) & (f <= 0.6)
        f_br = f[freq_mask]
        Sxx_br = Sxx[freq_mask, :]

        # time mask
        # time_mask = (t_spec >= t_spec[0]) & (t_spec <= t_spec[0] + 30 * 11)
        # Sxx_br = Sxx_br[:, time_mask]
        # t_spec = t_spec[time_mask]

        
        # Find dominant breathing frequency for each time window
        idx = np.argmax(Sxx_br, axis=0)
        br_axis = 60 * f_br[idx]
        breathing_rates.append(br_axis)
    
    # Combine axes (e.g., median or weighted average)
    BR_combined = np.median(breathing_rates, axis=0)
    BR_smooth = uniform_filter1d(BR_combined, size=7)
    
    return f_br, t_spec, Sxx_br, BR_smooth

def plot_breathing_results(f, t_spec, p, BR_smooth, save_to=None):
    plt.figure(figsize=(10, 6))
    extent = [t_spec[0], t_spec[-1], f[0] * 60, f[-1] * 60]
    plt.imshow(
        p,
        aspect="auto",
        origin="lower",
        extent=extent,
        alpha=0.4,
        cmap="plasma"  # Different colormap for breathing
    )
    plt.scatter(t_spec, BR_smooth, s=8, color="cyan", label="Motion Breathing Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Breathing Rate (breaths/min)")
    plt.title("Motion-Based Breathing Rate over Time")
    plt.ylim(5, 40)  # Typical breathing rate range
    
    plt.legend()
    plt.colorbar(label="Normalized Power Spectral Density")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()

def analyze_accel_breathing(accel_df, time_resolution=60, save_to=None):
    """
    Analyze accelerometer data to extract breathing rate.
    
    Parameters:
    -----------
    accel_df : pandas.DataFrame
        DataFrame with columns: time, x, y, z
    time_resolution : int, default=60
        Time window in seconds for spectrogram analysis
    save_to : str or Path, optional
        File path to save the plot
    
    Returns:
    --------
    dict : Analysis results containing breathing rate and spectral data
    """
    t, x, y, z = preprocess_accel(accel_df)
    # f, t_spec, p, BR_smooth, width, fs = compute_motion_breathing_rate(
    #     t, x, y, z, time_resolution
    # )
    f, t_spec, p, BR_smooth = compute_breathing_multiaxis(
        t, x, y, z, time_resolution
    )
    
    plot_breathing_results(f, t_spec, p, BR_smooth, save_to=save_to)
    
    print(f"Mean breathing rate: {np.nanmean(BR_smooth):.1f} breaths/min")
    print(f"Breathing rate std: {np.nanstd(BR_smooth):.1f} breaths/min")
    
    return {
        "f": f,
        "t_spec": t_spec,
        "p": p,
        "BR_smooth": BR_smooth,
    }

def create_breathing_signal(t_spec, BR_smooth, fs_output=10):
    """
    Create a synthetic breathing signal from extracted breathing rates.
    
    Parameters:
    -----------
    t_spec : array
        Time points from spectrogram
    BR_smooth : array  
        Breathing rates in breaths/min
    fs_output : float
        Sampling frequency for output signal
        
    Returns:
    --------
    t_breathing : array
        Time vector for breathing signal
    breathing_signal : array
        Synthetic breathing waveform
    """
    # Create high-resolution time vector
    t_start, t_end = t_spec[0], t_spec[-1]
    t_breathing = np.arange(t_start, t_end, 1/fs_output)
    
    # Interpolate breathing rate to high-resolution time grid
    BR_interp = np.interp(t_breathing, t_spec, BR_smooth)
    
    # Convert breathing rate to instantaneous frequency (Hz)
    # freq_hz = Breaths / second
    # period = 1 / freq_hz
    freq_hz = BR_interp / 60
    
    # Create phase by integrating frequency
    # Phase = 2π * ∫ frequency dt
    dt = 1/fs_output
    phase = 2 * np.pi * np.cumsum(freq_hz) * dt
    # sin(P * x) has period 2*np.pi/P, so we need to scale the frequency
    # P = 2 * np.pi / freq_hz
    # dynamic_scalar = 2 * np.pi / freq_hz
    
    # Generate sinusoidal breathing signal
    # breathing_signal = np.sin(dynamic_scalar * t_breathing)
    breathing_signal = np.sin(phase)
    
    return t_breathing, breathing_signal

def plot_accel_axes_with_breathing(t, x, y, z, t_spec, BR_smooth, save_to=None):
    """
    Plot individual accelerometer axes and reconstructed breathing signal.
    
    Parameters:
    -----------
    t : array
        Time vector for accelerometer data
    x, y, z : arrays
        Accelerometer data for each axis
    t_spec : array
        Time points from spectrogram
    BR_smooth : array
        Breathing rates in breaths/min
    save_to : str or Path, optional
        File path to save the plot
    """
    # Create synthetic breathing signal
    t_breathing, breathing_signal = create_breathing_signal(t_spec, BR_smooth)

    # limit the x,y,z to the same time range as t_spec
    start_idx = np.searchsorted(t, t_spec[0])
    end_idx = np.searchsorted(t, t_spec[-1])
    t = t[start_idx:end_idx]
    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    z = z[start_idx:end_idx]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    
    # Colors matching the reference image
    colors = ['#4A90E2', '#E24A4A', '#4AE24A', '#00CED1']
    
    # Plot X axis
    axes[0].plot(t, x, color=colors[0], linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel('Accel X', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(np.percentile(x, [1, 99]))
    
    # Plot Y axis
    axes[1].plot(t, y, color=colors[1], linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel('Accel Y', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(np.percentile(y, [1, 99]))
    
    # Plot Z axis
    axes[2].plot(t, z, color=colors[2], linewidth=0.8, alpha=0.8)
    axes[2].set_ylabel('Accel Z', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(np.percentile(z, [1, 99]))
    
    # Plot reconstructed breathing signal
    axes[3].plot(t_breathing, breathing_signal, color=colors[3], linewidth=1.2, alpha=0.9)
    axes[3].set_ylabel('Inferred\nBreathing', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-1.2, 1.2)
    
    # Add breathing rate annotations
    for i in range(0, len(t_spec), max(1, len(t_spec)//5)):  # Show ~5 annotations
        rate = BR_smooth[i]
        axes[3].annotate(f'{rate:.1f} br/min', 
                        xy=(t_spec[i], 1.0), 
                        xytext=(t_spec[i], 1.3),
                        ha='center', va='bottom', fontsize=9,
                        alpha=0.7,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Set common properties
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F9FA')
    
    # Set overall plot properties
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    
    # Add labels A, B, C, D
    labels = ['A', 'B', 'C', 'D']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()

def plot_accel_axes_with_breathing_interactive(t, x, y, z, t_spec, BR_smooth, save_to=None):
    """
    Plot individual accelerometer axes and reconstructed breathing signal with synchronized zooming.
    
    Parameters:
    -----------
    t : array
        Time vector for accelerometer data
    x, y, z : arrays
        Accelerometer data for each axis
    t_spec : array
        Time points from spectrogram
    BR_smooth : array
        Breathing rates in breaths/min
    save_to : str or Path, optional
        File path to save the plot
    """
    # Create synthetic breathing signal
    t_breathing, breathing_signal = create_breathing_signal(t_spec, BR_smooth)

    # limit the x,y,z to the same time range as t_spec
    start_idx = np.searchsorted(t, t_spec[0])
    end_idx = np.searchsorted(t, t_spec[-1])
    t = t[start_idx:end_idx]
    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    z = z[start_idx:end_idx]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 5), sharex=True)
    
    # Colors matching the reference image
    colors = ['#4A90E2', '#E24A4A', '#4AE24A', '#00CED1']
    
    # Store original data and limits
    original_data = {'t': t, 'x': x, 'y': y, 'z': z}
    original_xlim = (t[0], t[-1])
    original_ylims = {
        0: (np.percentile(x, [1, 99])),
        1: (np.percentile(y, [1, 99])),
        2: (np.percentile(z, [1, 99])),
        3: (-1.2, 1.2)
    }
    
    # Plot X axis
    line_x, = axes[0].plot(t, x, color=colors[0], linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel('Accel X', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(original_ylims[0])
    
    # Plot Y axis
    line_y, = axes[1].plot(t, y, color=colors[1], linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel('Accel Y', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(original_ylims[1])
    
    # Plot Z axis
    line_z, = axes[2].plot(t, z, color=colors[2], linewidth=0.8, alpha=0.8)
    axes[2].set_ylabel('Accel Z', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(original_ylims[2])
    
    # Plot reconstructed breathing signal
    line_breathing, = axes[3].plot(t_breathing, breathing_signal, color=colors[3], linewidth=1.2, alpha=0.9)
    axes[3].set_ylabel('Inferred\nBreathing', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(original_ylims[3])
    
    # Add breathing rate annotations (initially for full range)
    rate_annotations = []
    for i in range(0, len(t_spec), max(1, len(t_spec)//5)):
        rate = BR_smooth[i]
        ann = axes[3].annotate(f'{rate:.1f} br/min', 
                              xy=(t_spec[i], 1.0), 
                              xytext=(t_spec[i], 1.3),
                              ha='center', va='bottom', fontsize=9,
                              alpha=0.7,
                              arrowprops=dict(arrowstyle='->', alpha=0.5))
        rate_annotations.append(ann)
    
    # Function to rescale Y-axis based on visible data
    def rescale_yaxis(xmin, xmax):
        """Rescale Y-axis for each accelerometer axis based on visible time range."""
        # Find indices corresponding to the time range
        mask = (t >= xmin) & (t <= xmax)
        
        if np.any(mask):
            # Get data within the time range
            x_visible = x[mask]
            y_visible = y[mask]
            z_visible = z[mask]
            
            # Calculate new Y-limits with some padding
            def get_ylims_with_padding(data, padding_factor=0.1):
                if len(data) > 0:
                    data_min, data_max = np.min(data), np.max(data)
                    data_range = data_max - data_min
                    padding = data_range * padding_factor
                    return (data_min - padding, data_max + padding)
                else:
                    return (0, 1)  # fallback
            
            # Set new Y-limits for accelerometer axes
            axes[0].set_ylim(get_ylims_with_padding(x_visible))
            axes[1].set_ylim(get_ylims_with_padding(y_visible))
            axes[2].set_ylim(get_ylims_with_padding(z_visible))
            
            # Breathing signal Y-axis stays fixed at (-1.2, 1.2)
            # axes[3] keeps its original limits
    
    # Function to update annotations based on current view
    def update_annotations():
        # Clear existing annotations
        for ann in rate_annotations:
            ann.remove()
        rate_annotations.clear()
        
        # Get current x-axis limits
        xlim = axes[3].get_xlim()
        
        # Find t_spec points within current view
        visible_mask = (t_spec >= xlim[0]) & (t_spec <= xlim[1])
        visible_t_spec = t_spec[visible_mask]
        visible_br = BR_smooth[visible_mask]
        
        # Add annotations for visible points (max 10 to avoid cluttering)
        step = max(1, len(visible_t_spec) // 10)
        for i in range(0, len(visible_t_spec), step):
            rate = visible_br[i]
            ann = axes[3].annotate(f'{rate:.1f} br/min', 
                                  xy=(visible_t_spec[i], 1.0), 
                                  xytext=(visible_t_spec[i], 1.3),
                                  ha='center', va='bottom', fontsize=9,
                                  alpha=0.7,
                                  arrowprops=dict(arrowstyle='->', alpha=0.5))
            rate_annotations.append(ann)
        
        fig.canvas.draw()
    
    # Function to handle span selection (zoom)
    def onselect(xmin, xmax):
        # Set new x-axis limits for all subplots
        for ax in axes:
            ax.set_xlim(xmin, xmax)
        
        # Rescale Y-axis for accelerometer data
        rescale_yaxis(xmin, xmax)
        
        # Update annotations
        update_annotations()
        
        fig.canvas.draw()
    
    # Function to reset zoom
    def reset_zoom(event):
        if event.key == 'r':
            # Reset X-axis
            for ax in axes:
                ax.set_xlim(original_xlim)
            
            # Reset Y-axis to original limits
            for i, ax in enumerate(axes):
                ax.set_ylim(original_ylims[i])
            
            update_annotations()
            fig.canvas.draw()
    
    # Add span selectors to each subplot for synchronized zooming
    span_selectors = []
    for ax in axes:
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                           props=dict(alpha=0.3, facecolor='yellow'))
        span_selectors.append(span)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', reset_zoom)
    
    # Set common properties
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F9FA')
    
    # Add labels A, B, C, D
    labels = ['A', 'B', 'C', 'D']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add instruction text
    fig.suptitle('Interactive Accelerometer Data Viewer\n'
                'Click and drag to zoom on any axis (syncs all axes). Press "r" to reset zoom.\n'
                'Y-axis auto-scales for accelerometer data based on visible range.',
                fontsize=14, y=0.98)
    
    # Set overall plot properties
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, top=0.91)  # Make room for title
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()

def analyze_accel_breathing_with_axes_interactive(accel_df, time_resolution=60, save_to=None):
    """
    Analyze accelerometer data and create the interactive multi-axis plot with breathing signal.
    
    Parameters:
    -----------
    accel_df : pandas.DataFrame
        DataFrame with columns: time, x, y, z
    time_resolution : int, default=60
        Time window in seconds for spectrogram analysis
    save_to : str or Path, optional
        File path to save the plot
    
    Returns:
    --------
    dict : Analysis results containing breathing rate and synthetic signal
    """
    t, x, y, z = preprocess_accel(accel_df)
    f, t_spec, p, BR_smooth = compute_breathing_multiaxis(
        t, x, y, z, time_resolution
    )
    
    # Create the interactive multi-axis plot
    plot_accel_axes_with_breathing_interactive(t, x, y, z, t_spec, BR_smooth, save_to=save_to)
    
    print(f"Mean breathing rate: {np.nanmean(BR_smooth):.1f} breaths/min")
    print(f"Breathing rate std: {np.nanstd(BR_smooth):.1f} breaths/min")
    print("\nInteractive Controls:")
    print("- Click and drag on any axis to zoom (all axes sync)")
    print("- Press 'r' to reset zoom to full view")
    
    # Create synthetic breathing signal for return
    t_breathing, breathing_signal = create_breathing_signal(t_spec, BR_smooth)
    
    return {
        "f": f,
        "t_spec": t_spec,
        "p": p,
        "BR_smooth": BR_smooth,
        "t_breathing": t_breathing,
        "breathing_signal": breathing_signal,
    }

if __name__ == "__main__":
    from pathlib import Path
    from pisces.data_sets import DataSetObject
    # Example usage:
    # Load your accelerometer and heart rate data into pandas DataFrames
    # data_dir = Path("your_data_directory")  # replace with your data directory
    # data_dir = Path("/Users/eric/Engineering/Work/pisces/data/walch_et_al")  # replace with your data directory
    data_set_name = "walch_et_al"
    # data_set_name = "hybrid_motion"
    data_dir = Path(f"/Users/eric/Engineering/Work/pisces/data/{data_set_name}")  # replace with your data directory

    
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
            # analyze_accel_vs_hr(accel_df, hr_df,
            #                     time_resolution=30,
            #                     save_to=cwd / f"{data_set_name}_{idno}_accel_vs_hr.png")
            
            # Add breathing rate analysis
            # analyze_accel_breathing(accel_df,
            #                        time_resolution=60,
            #                        save_to=cwd / f"{data_set_name}_{idno}_breathing_rate.png")
            analyze_accel_breathing_with_axes_interactive(accel_df,
                                            time_resolution=60,
                                            # save_to=cwd / f"{data_set_name}_{idno}_breathing_axes.png")
                                            save_to=None
                                            )
            break
        except Exception as e:
            print(f"Error processing ID {idno}: {e}")
            raise e
# Example usage:
# accel_df = pd.read_csv("your_accel.csv")  # columns: time, x, y, z
# hr_df = pd.read_csv("your_hr.csv")        # columns: time, bpm
# analyze_accel_vs_hr(accel_df, hr_df)