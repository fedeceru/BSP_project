import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import welch
from matplotlib.transforms import offset_copy

RESULTS_DIR = '../results'

def _slugify(text):
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '_', text)

def _save_figure(fig, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')

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
    _save_figure(fig, f"{_slugify(title)}.png")
    plt.show()

def plot_before_after_comprehensive(t, before, after, fs, title="Pipeline Evaluation: BWR + PLI Cancellation",
                                    before_color='#0021fa', after_color='#fc0000',
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
    _save_figure(fig, f"{_slugify(title)}.png")
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
        ax.plot(t, display_channels[:, i] + (n_channels - i) * offset_step, color='#2c3e50', lw=0.8)

    ax.plot(t, display_enhanced, color="#0021fa", lw=1.2, label='Enhanced Signal (PCA)')

    if len(peaks) > 0:
        mask = (peaks >= 0) & (peaks < len(t))
        valid_peaks = peaks[mask]
        marker_size = 10
        # A 'v' marker's tip sits markersize/2 points below its anchor. Shift the
        # anchor up by that same amount (in points, not data units) so the tip
        # lands exactly on the peak regardless of the y-axis data scale.
        tip_transform = offset_copy(ax.transData, fig=fig, x=0, y=marker_size / 2, units='points')
        ax.plot(t[valid_peaks], display_enhanced[valid_peaks],
                marker='v', color='none', markeredgecolor="#fc0000", markersize=marker_size,
                linestyle='None', label='Detected QRS', transform=tip_transform)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_yticks([])
    ax.legend(loc='upper right')

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
    _save_figure(fig, f"{_slugify(suptitle)}.png")
    plt.show()

def plot_pipeline_stages(t, s1, s4, s5, s6, title="Sequential Analysis Pipeline"):
    # Shows S4 rather than S3 for the second panel: S3 is at the original
    # sampling rate and can't be sliced with a fs_target-domain time axis, so
    # S4 (the upsampled, otherwise-identical signal) stands in for it here.
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
    fig.suptitle(title, fontweight='bold', fontsize=14)

    n_channels = min(5, s1.shape[1])
    offset_step = np.max(np.abs(s4)) * 2.5

    signals = [s1, s4, s5]
    titles = ['S1 (Raw)', 'S4 (BWR+PLI+Upsampled)', 'S5 (MECG Removed)']

    for col, (sig, t_title) in enumerate(zip(signals, titles)):
        for ch in range(n_channels):
            axes[col].plot(t, sig[:, ch] + (n_channels - ch) * offset_step, color='black', lw=0.8)
        axes[col].set_title(t_title)
        axes[col].set_xlabel('Time [s]')
        axes[col].set_xticks([])

    t_avg = np.linspace(-0.125, 0.125, len(s6))
    for ch in range(n_channels):
        axes[3].plot(t_avg, s6[:, ch] + (n_channels - ch) * offset_step, color='#fc0000', lw=1.5)
    axes[3].set_title('S6 (Avg FECG)')
    axes[3].set_xlabel('Time [s]')

    axes[0].set_yticks([])
    plt.tight_layout()
    _save_figure(fig, f"{_slugify(title)}.png")
    plt.show()

def plot_ica_stages(t, s1, s3_or_s4, ica_sources, best_fecg_avg, title="ICA Algorithm Pipeline"):
    # Shows S4 rather than S3 for the second panel: S3 is at the original
    # sampling rate and can't be sliced with a fs_target-domain time axis, so
    # S4 (the upsampled, otherwise-identical signal) stands in for it here.
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=False)
    fig.suptitle(title, fontweight='bold', fontsize=14)

    n_channels = min(10, s1.shape[1])
    n_sources = min(10, ica_sources.shape[1])

    offset_sig = np.max(np.abs(s3_or_s4)) * 2.5
    for ch in range(n_channels):
        axes[0].plot(t, s1[:, ch] + (n_channels - ch) * offset_sig, color='black', lw=0.8)
        axes[1].plot(t, s3_or_s4[:, ch] + (n_channels - ch) * offset_sig, color='black', lw=0.8)
    axes[0].set_title('Unipolar Signals (X)')
    axes[1].set_title('Filtered Signals')
    
    offset_ica = np.max(np.abs(ica_sources)) * 2.5
    for ch in range(n_sources):
        axes[2].plot(t, ica_sources[:, ch] + (n_sources - ch) * offset_ica, color='#0021fa', lw=0.8)
    axes[2].set_title('ICA Sources (S)')
    
    t_avg = np.linspace(-0.125, 0.125, len(best_fecg_avg))
    axes[3].plot(t_avg, best_fecg_avg, color='#fc0000', lw=1.5)
    axes[3].set_title('Avg FECG (Best Source)')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Time [s]')
        
    plt.tight_layout()
    _save_figure(fig, f"{_slugify(title)}.png")
    plt.show()

def plot_fhr_traces(t_sa, fhr_sa, t_ica, fhr_ica):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    
    ax1.plot(t_sa, fhr_sa, marker='.', color='#2c3e50', linestyle='None', markersize=4)
    ax1.set_title("FHR Trace - SA", fontweight='bold')
    ax1.set_ylabel("FHR [bpm]")
    
    ax2.plot(t_ica, fhr_ica, marker='.', color='#16a085', linestyle='None', markersize=4)
    ax2.set_title("FHR Trace - ICA", fontweight='bold')
    ax2.set_ylabel("FHR [bpm]")
    ax2.set_xlabel("Time [s]")
    
    ax1.set_ylim([50, 220])
    ax1.set_xlim([0, 60])

    plt.tight_layout()
    _save_figure(fig, "fhr_traces.png")
    plt.show()

def plot_detection_validation(t_zoom, channels_zoom, detected_idx, gt_idx,
                              t_avg_ms, avg_beat, n_beats):
    channel_colors = ['#2c3e50', '#16a085', '#8e44ad', '#f39c12']
    n_channels = min(len(channel_colors), channels_zoom.shape[1])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10))

    # Joyplot/waterfall stack: offset each channel's baseline by an amount
    # derived from the 2nd/98th percentile spread (robust to isolated QRS
    # outliers, unlike max(abs(...))), with headroom so traces don't collide.
    p_lo, p_hi = np.percentile(channels_zoom[:, :n_channels], [2, 98], axis=0)
    channel_spread = p_hi - p_lo
    offset_step = channel_spread.max() * 3 if channel_spread.max() > 0 else 2

    for ch in range(n_channels):
        offset = (n_channels - ch) * offset_step
        ax_top.plot(t_zoom, channels_zoom[:, ch] + offset, color=channel_colors[ch], lw=0.6)

    for ch in range(n_channels):
        offset = (n_channels - ch) * offset_step
        if len(detected_idx) > 0:
            ax_top.plot(t_zoom[detected_idx], channels_zoom[detected_idx, ch] + offset,
                        marker='o', linestyle='None', color='#e74c3c', 
                        markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=3,
                        label='Detected (SA)' if ch == 0 else None)

    # Ground Truth: centre line + tolerance band
    tolerance_s = 0.05  # Example: 50 ms tolerance window
    for i, idx in enumerate(gt_idx):
        t_gt = t_zoom[idx]

        # Centre line
        ax_top.axvline(t_gt, color='#2980b9', lw=1.2, alpha=0.7, zorder=1,
                       label='Ground Truth' if i == 0 else None)

        # Tolerance band
        ax_top.axvspan(t_gt - tolerance_s, t_gt + tolerance_s,
                       color='#2980b9', alpha=0.1, zorder=0,
                       label='Tolerance Window' if i == 0 else None)

    ax_top.set_title("Multi-channel Signal with Detection Validation", fontweight='bold')
    ax_top.set_xlabel('Time (s)')
    ax_top.set_yticks([(n_channels - ch) * offset_step for ch in range(n_channels)])
    ax_top.set_yticklabels([f'Ch {ch + 1}' for ch in range(n_channels)])
    
    # Remove the vertical grid so it isn't confused with the GT markers
    ax_top.xaxis.grid(False)

    # Move the legend outside the plot if it covers the signals, or use a 3-column layout
    ax_top.legend(loc='upper right', ncol=3, fontsize=9, framealpha=0.9)

    for ch in range(n_channels):
        ax_bottom.plot(t_avg_ms, avg_beat[:, ch], color=channel_colors[ch], lw=1.8, label=f'Ch {ch + 1}')

    ax_bottom.set_title(f"Average Fetal Morphology (Multi-channel) – Based on {n_beats} Beats", fontweight='bold')
    ax_bottom.set_xlabel('Time (ms)')
    ax_bottom.set_ylabel('Mean Amplitude')
    ax_bottom.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(fig, "detection_validation.png")
    plt.show()

def plot_performance_summary(df_results, success_rate_sa, success_rate_ica):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    ax1.bar(['SA', 'ICA'], [success_rate_sa, success_rate_ica],
            color=['#2c3e50', '#16a085'])
    ax1.set_ylabel('FHR detection success rate [%]')
    ax1.set_ylim([0, 100])
    ax1.set_title('Success Rate', fontweight='bold')

    # Reliability distribution (feasible patients only)
    rel_sa = df_results.loc[df_results['feasible_sa'], 'reliability_sa']
    rel_ica = df_results.loc[df_results['feasible_ica'], 'reliability_ica']
    ax2.boxplot([rel_sa, rel_ica], tick_labels=['SA', 'ICA'])
    ax2.set_ylabel('FHR detection reliability')
    ax2.set_ylim([0, 1.05])
    ax2.set_title('Reliability', fontweight='bold')

    fig.suptitle('FHR Detection Performance', fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save_figure(fig, "performance_summary.png")
    plt.show()

def plot_ground_truth_validation(prec_sa, rec_sa, f1_sa, prec_ica, rec_ica, f1_ica, tolerance_ms):
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ['Precision', 'Recall', 'F1']
    sa_vals = [prec_sa * 100, rec_sa * 100, f1_sa * 100]
    ica_vals = [prec_ica * 100, rec_ica * 100, f1_ica * 100]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, sa_vals, width, label='SA', color='#2c3e50')
    ax.bar(x + width / 2, ica_vals, width, label='ICA', color='#16a085')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('%')
    ax.set_ylim([0, 105])
    ax.set_title(f'Fetal QRS Detection vs Ground Truth (±{tolerance_ms:.0f} ms tolerance)', fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(fig, "ground_truth_validation.png")
    plt.show()

def _add_trend_line(ax, x, y, color='red'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 2:
        return
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    r = np.corrcoef(x[valid], y[valid])[0, 1]
    x_line = np.array([x[valid].min(), x[valid].max()])
    sign = '+' if intercept >= 0 else '-'
    label = f'Linear Fit\n(y={slope:.2f}x {sign} {abs(intercept):.2f})\nR={r:.2f}'
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, label=label)
    ax.legend(loc='best', fontsize=8)

def plot_snr_sir_vs_reliability(df_fig10):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    axes[0, 0].scatter(df_fig10['mean_snr_db'], df_fig10['reliability_sa_fig10'],
                       marker='x', color="#222d37", alpha=0.95)
    _add_trend_line(axes[0, 0], df_fig10['mean_snr_db'], df_fig10['reliability_sa_fig10'])
    axes[0, 0].set_ylabel('SA reliability')
    axes[0, 0].set_title('SNR', fontweight='bold')

    axes[0, 1].scatter(df_fig10['mean_sir_db'], df_fig10['reliability_sa_fig10'],
                       marker='x', color="#222d37", alpha=0.95)
    axes[0, 1].set_title('SIR', fontweight='bold')

    axes[1, 0].scatter(df_fig10['mean_snr_db'], df_fig10['reliability_ica_fig10'],
                       marker='x', color="#1f8a75", alpha=0.95)
    _add_trend_line(axes[1, 0], df_fig10['mean_snr_db'], df_fig10['reliability_ica_fig10'])
    axes[1, 0].set_xlabel('Mean SNR [dB]')
    axes[1, 0].set_ylabel('ICA reliability')

    axes[1, 1].scatter(df_fig10['mean_sir_db'], df_fig10['reliability_ica_fig10'],
                       marker='x', color="#1f8a75", alpha=0.95)
    axes[1, 1].set_xlabel('Mean SIR [dB]')

    for ax in axes.flat:
        ax.set_ylim([-0.05, 1.05])

    fig.suptitle('FHR Detection Reliability vs SNR/SIR', fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save_figure(fig, "snr_sir_vs_reliability.png")
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_reliability_correlation_matrix(df_fig10):
    cols = ['reliability_sa_fig10', 'reliability_ica_fig10', 'mean_snr_db', 'mean_sir_db']
    labels = ['SA Reliability', 'ICA Reliability', 'SNR', 'SIR']
    corr_matrix = df_fig10[cols].corr(method='spearman').values
    n = len(labels)
    colors = ['#2c3e50', '#ffffff', '#16a085']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(corr_matrix, cmap=custom_cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text_color = "black"
            
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color)

    ax.set_title("Spearman Correlation Matrix", fontweight='bold', pad=15)
    fig.colorbar(cax, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    _save_figure(fig, "reliability_correlation_matrix.png")
    plt.show()

def plot_tolerance_sensitivity(df_tolerance):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_tolerance['tolerance_ms'], df_tolerance['f1_sa'] * 100, marker='o',
            color='#2c3e50', label='SA')
    ax.plot(df_tolerance['tolerance_ms'], df_tolerance['f1_ica'] * 100, marker='o',
            color='#16a085', label='ICA')

    ax.set_xlabel('Ground-truth matching tolerance [ms]')
    ax.set_ylabel('F1 score [%]')
    ax.set_ylim([0, 105])
    ax.set_title('Fetal QRS Detection F1 vs Ground-Truth Matching Tolerance', fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    _save_figure(fig, "tolerance_sensitivity.png")
    plt.show()

def plot_fhr_accuracy(gt_sa, est_sa, mae_sa, rmse_sa, gt_ica, est_ica, mae_ica, rmse_ica):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    lims = [60, 220]
    for ax, gt_vals, est_vals, mae, rmse, label, color in (
        (ax1, gt_sa, est_sa, mae_sa, rmse_sa, 'SA', '#2c3e50'),
        (ax2, gt_ica, est_ica, mae_ica, rmse_ica, 'ICA', '#16a085'),
    ):
        ax.plot(lims, lims, color='#999999', linestyle='--', linewidth=1, label='Identity (y = x)')
        ax.scatter(gt_vals, est_vals, s=10, alpha=0.3, color=color)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Ground-truth FHR [bpm]')
        ax.set_title(f'{label}\nMAE: {mae:.1f} bpm   RMSE: {rmse:.1f} bpm', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    ax1.set_ylabel('Estimated FHR [bpm]')

    fig.suptitle('Estimated vs Ground-Truth FHR (all records, 1s grid)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save_figure(fig, "fhr_accuracy.png")
    plt.show()