import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import welch
from matplotlib.transforms import offset_copy

plt.rcParams.update({
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 10,
    'legend.frameon': True,
    'legend.facecolor': 'white'
})

def plot_filter_transfer_function(b, a, fs, title="Baseline Wander Remover (FIR Filter)"):
    w, h = signal.freqz(b, a, worN=8000)
    freq = (w * fs) / (2 * np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(freq, 20 * np.log10(abs(h) + 1e-10), color='#2c3e50')
    ax1.set_title("Magnitude", fontweight='bold')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_xlim([0, 50])
    ax1.set_ylim([-40, 10])

    angles = np.unwrap(np.angle(h))
    ax2.plot(freq, np.degrees(angles), color='#2c3e50')
    ax2.set_title("Phase", fontweight='bold')
    ax2.set_ylabel('Phase [degrees]')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_xlim([0, 50])

    plt.suptitle(title, fontweight='bold', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_before_after(t, before, after, title="Before / After", before_label="Before", after_label="After"):
    n_channels = min(5, before.shape[1])
    offset_step = np.max(np.abs(before)) * 2.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    for ch in range(n_channels):
        ax1.plot(t, before[:, ch] + (n_channels - ch) * offset_step, color='#333333', lw=0.8)
        ax2.plot(t, after[:, ch] + (n_channels - ch) * offset_step, color='#2c3e50', lw=0.8)

    ax1.set_title(before_label, fontweight='bold')
    ax2.set_title(after_label, fontweight='bold')
    ax1.set_xlabel('Time [s]')
    ax2.set_xlabel('Time [s]')
    ax1.set_yticks([])

    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_before_after_comprehensive(t, before, after, fs, title="Pipeline Evaluation: BWR + PLI Cancellation",
                                    before_color='purple', after_color='green',
                                    before_label="S1 (Raw)", after_label="S3 (Filtered)",
                                    show_psd=True):
    n_channels = min(4, before.shape[1]) # Limit to 4 channels for clarity
    step = np.max(np.abs(before)) * 2.5
    group_step = step * 2.3

    if show_psd:
        fig = plt.figure(figsize=(14, 2.0 * n_channels), layout='constrained')
        # Grid: 70% of the space for time, 30% for frequency
        gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15)
        ax_time = fig.add_subplot(gs[0])
        ax_freq = fig.add_subplot(gs[1])
    else:
        fig, ax_time = plt.subplots(figsize=(10, 2.0 * n_channels), layout='constrained')
        ax_freq = None

    for ch in range(n_channels):
        base = (n_channels - 1 - ch) * group_step

        ax_time.plot(t, before[:, ch] + base + step, color=before_color, lw=0.9,
                     label=before_label if ch == 0 else None)
        ax_time.plot(t, after[:, ch] + base, color=after_color, lw=0.9,
                     label=after_label if ch == 0 else None)

        if show_psd:
            f_before, psd_before = welch(before[:, ch], fs, nperseg=1024)
            f_after, psd_after = welch(after[:, ch], fs, nperseg=1024)

            freq_base = (n_channels - 1 - ch) * 50  # Arbitrary offset for visualisation
            ax_freq.plot(f_before, 10*np.log10(psd_before) + freq_base, color=before_color, lw=0.8)
            ax_freq.plot(f_after, 10*np.log10(psd_after) + freq_base, color=after_color, lw=0.8)

    # Time axis formatting
    ax_time.set_yticks([])
    ax_time.set_xlabel('Time [s]')
    ax_time.legend(loc='upper right')
    if show_psd:
        ax_time.set_title("Time Domain", fontweight='bold')

        # Frequency axis formatting
        ax_freq.set_yticks([])
        ax_freq.set_xlabel('Frequency [Hz]')
        ax_freq.set_xlim(0, 100) # Limit to 100Hz to clearly see the 50/60Hz components
        ax_freq.set_title("Power Spectral Density", fontweight='bold')

    fig.suptitle(title, fontweight='bold', fontsize=16)
    plt.show()

def _draw_qrs_detection_panel(ax, fig, t, channels_matrix, enhanced_signal, peaks, title, equalize_channels=False):
    n_channels = min(5, channels_matrix.shape[1])
    display_channels = channels_matrix[:, :n_channels]
    display_enhanced = enhanced_signal

    if equalize_channels:
        # Display-only rescaling: normalise each lead to the same amplitude so
        # complexes on quieter channels are not dwarfed by louder ones. Purely
        # cosmetic -- detection runs on the original, unscaled signal.
        channel_scales = np.max(np.abs(display_channels), axis=0)
        channel_scales[channel_scales == 0] = 1.0
        target_amp = np.median(channel_scales)
        display_channels = display_channels / channel_scales * target_amp
        offset_step = target_amp * 2.5

        # Also rescale the enhanced (PCA) trace to that same target amplitude --
        # on its own it is much flatter than the raw leads and the complexes are
        # hard to see. Detection already ran on the unscaled signal.
        enhanced_amp = np.max(np.abs(enhanced_signal))
        if enhanced_amp > 0:
            display_enhanced = enhanced_signal / enhanced_amp * target_amp
    else:
        offset_step = np.max(np.abs(channels_matrix)) * 1.5

    for i in range(n_channels):
        ax.plot(t, display_channels[:, i] + (n_channels - i) * offset_step, color='#333333', lw=0.8)

    ax.plot(t, display_enhanced, color='#2980b9', lw=1.2, label='Enhanced Signal (PCA)')

    if len(peaks) > 0:
        mask = (peaks >= 0) & (peaks < len(t))
        valid_peaks = peaks[mask]
        marker_size = 10
        # A 'v' marker's tip sits markersize/2 points below its anchor. Shift the
        # anchor up by that same amount (in points, not data units) so the tip
        # lands exactly on the peak regardless of the y-axis data scale.
        tip_transform = offset_copy(ax.transData, fig=fig, x=0, y=marker_size / 2, units='points')
        ax.plot(t[valid_peaks], display_enhanced[valid_peaks],
                marker='v', color='none', markeredgecolor='#e74c3c', markersize=marker_size,
                linestyle='None', label='Detected QRS', transform=tip_transform)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_yticks([])
    ax.legend(loc='upper right')

def plot_qrs_detection(t, channels_matrix, enhanced_signal, peaks, title="QRS Detection"):
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_qrs_detection_panel(ax, fig, t, channels_matrix, enhanced_signal, peaks, title)
    plt.tight_layout()
    plt.show()

def plot_qrs_detection_dual(t, maternal_channels, maternal_enhanced, maternal_peaks,
                            fetal_channels, fetal_enhanced, fetal_peaks,
                            maternal_title="Maternal QRS Detection (S4)",
                            fetal_title="Fetal QRS Detection (S5)",
                            suptitle="Maternal and Fetal QRS Detection"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    _draw_qrs_detection_panel(ax1, fig, t, maternal_channels, maternal_enhanced, maternal_peaks, maternal_title,
                              equalize_channels=True)
    _draw_qrs_detection_panel(ax2, fig, t, fetal_channels, fetal_enhanced, fetal_peaks, fetal_title)
    fig.suptitle(suptitle, fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_pipeline_stages(t, s1, s3, s5, s6, title="Sequential Analysis Pipeline"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
    fig.suptitle(title, fontweight='bold', fontsize=14)
    
    n_channels = min(5, s1.shape[1])
    offset_step = np.max(np.abs(s3)) * 2.5
    
    signals = [s1, s3, s5]
    titles = ['S1 (Raw)', 'S3 (BWR+PLC)', 'S5 (MECG Removed)']
    
    for col, (sig, t_title) in enumerate(zip(signals, titles)):
        for ch in range(n_channels):
            axes[col].plot(t, sig[:, ch] + (n_channels - ch) * offset_step, color='black', lw=0.8)
        axes[col].set_title(t_title)
        axes[col].set_xlabel('Time [s]')
        axes[col].set_xticks([])
        
    t_avg = np.linspace(-0.125, 0.125, len(s6))
    for ch in range(n_channels):
        axes[3].plot(t_avg, s6[:, ch] + (n_channels - ch) * offset_step, color='#8e44ad', lw=1.5)
    axes[3].set_title('S6 (Avg FECG)')
    axes[3].set_xlabel('Time [s]')
    
    axes[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_ica_stages(t, s1, s3, ica_sources, best_fecg_avg, title="ICA Algorithm Pipeline"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=False)
    fig.suptitle(title, fontweight='bold', fontsize=14)
    
    n_channels = min(10, s1.shape[1])
    n_sources = min(10, ica_sources.shape[1])
    
    offset_sig = np.max(np.abs(s3)) * 2.5
    for ch in range(n_channels):
        axes[0].plot(t, s1[:, ch] + (n_channels - ch) * offset_sig, color='black', lw=0.8)
        axes[1].plot(t, s3[:, ch] + (n_channels - ch) * offset_sig, color='black', lw=0.8)
    axes[0].set_title('Unipolar Signals (X)')
    axes[1].set_title('Filtered Signals')
    
    offset_ica = np.max(np.abs(ica_sources)) * 2.5
    for ch in range(n_sources):
        axes[2].plot(t, ica_sources[:, ch] + (n_sources - ch) * offset_ica, color='#16a085', lw=0.8)
    axes[2].set_title('ICA Sources (S)')
    
    t_avg = np.linspace(-0.125, 0.125, len(best_fecg_avg))
    axes[3].plot(t_avg, best_fecg_avg, color='#8e44ad', lw=1.5)
    axes[3].set_title('Avg FECG (Best Source)')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Time [s]')
        
    plt.tight_layout()
    plt.show()

def plot_fhr_traces(t_sa, fhr_sa, t_ica, fhr_ica):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    
    ax1.plot(t_sa, fhr_sa, marker='.', color='black', linestyle='None', markersize=4)
    ax1.set_title("FHR Trace - Sequential Analysis", fontweight='bold')
    ax1.set_ylabel("FHR [bpm]")
    
    ax2.plot(t_ica, fhr_ica, marker='.', color='#2c3e50', linestyle='None', markersize=4)
    ax2.set_title("FHR Trace - ICA", fontweight='bold')
    ax2.set_ylabel("FHR [bpm]")
    ax2.set_xlabel("Time [s]")
    
    ax1.set_ylim([50, 200])
    ax1.set_xlim([0, 60])
    
    plt.tight_layout()
    plt.show()

def plot_detection_validation(t_zoom, channels_zoom, detected_idx, gt_idx,
                              t_avg_ms, avg_beat, n_beats, record_label=""):
    channel_colors = ['#34495e', '#16a085', '#8e44ad', '#f39c12']
    n_channels = min(len(channel_colors), channels_zoom.shape[1])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10))

    for ch in range(n_channels):
        ax_top.plot(t_zoom, channels_zoom[:, ch], color=channel_colors[ch], lw=0.6, label=f'Ch {ch + 1}')

    amp_range = np.max(np.abs(channels_zoom)) if channels_zoom.size else 1.0
    marker_offset = amp_range * 0.12

    def _marker_heights(idx):
        heights = []
        for p in idx:
            lo, hi = max(0, p - 2), min(len(t_zoom), p + 3)
            heights.append(np.max(channels_zoom[lo:hi, :n_channels]) + marker_offset)
        return np.array(heights)

    if len(detected_idx) > 0:
        ax_top.plot(t_zoom[detected_idx], _marker_heights(detected_idx),
                    marker='v', linestyle='None', color='none', markeredgecolor='#e74c3c',
                    markeredgewidth=1.6, markersize=9, label='Detected (Algorithm)')

    if len(gt_idx) > 0:
        ax_top.plot(t_zoom[gt_idx], _marker_heights(gt_idx) + marker_offset * 0.6,
                    marker='D', linestyle='None', color='none', markeredgecolor='#27ae60',
                    markeredgewidth=1.6, markersize=8, label='Ground Truth')

    ax_top.set_title(f"Multi-channel Signal (10 s Zoom) – {record_label}", fontweight='bold')
    ax_top.set_xlabel('Time (s)')
    ax_top.set_ylabel('Amplitude')
    ax_top.legend(loc='upper right', ncol=2, fontsize=9)

    for ch in range(n_channels):
        ax_bottom.plot(t_avg_ms, avg_beat[:, ch], color=channel_colors[ch], lw=1.8, label=f'Ch {ch + 1}')

    ax_bottom.set_title(f"Average Fetal Morphology (Multi-channel) – Based on {n_beats} Beats", fontweight='bold')
    ax_bottom.set_xlabel('Time (ms)')
    ax_bottom.set_ylabel('Mean Amplitude')
    ax_bottom.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_performance_summary(df_results, success_rate_sa, success_rate_ica):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    ax1.bar(['Sequential\nAnalysis', 'ICA'], [success_rate_sa, success_rate_ica],
            color=['#2c3e50', '#16a085'])
    ax1.set_ylabel('FHR detection success rate [%]')
    ax1.set_ylim([0, 100])
    ax1.set_title('Success Rate', fontweight='bold')

    # Reliability distribution (feasible patients only)
    rel_sa = df_results.loc[df_results['feasible_sa'], 'reliability_sa']
    rel_ica = df_results.loc[df_results['feasible_ica'], 'reliability_ica']
    ax2.boxplot([rel_sa, rel_ica], labels=['Sequential\nAnalysis', 'ICA'])
    ax2.set_ylabel('FHR detection reliability')
    ax2.set_ylim([0, 1.05])
    ax2.set_title('Reliability', fontweight='bold')

    fig.suptitle('FHR Detection Performance (Section 2.4.1)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()