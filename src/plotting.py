import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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

def plot_qrs_detection(t, channels_matrix, enhanced_signal, peaks, title="QRS Detection"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_channels = min(5, channels_matrix.shape[1])
    offset_step = np.max(np.abs(channels_matrix)) * 1.5
    
    for i in range(n_channels):
        ax.plot(t, channels_matrix[:, i] + (n_channels - i) * offset_step, color='#333333', lw=0.8)
        
    ax.plot(t, enhanced_signal, color='#d35400', lw=1.2, label='Enhanced Signal (PCA)')
    
    if len(peaks) > 0:
        mask = (peaks < len(t))
        valid_peaks = peaks[mask]
        ax.plot(t[valid_peaks], enhanced_signal[valid_peaks] + (offset_step*0.2), 
                marker='v', color='none', markeredgecolor='#2980b9', markersize=10, 
                linestyle='None', label='Detected QRS')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_yticks([]) 
    ax.legend(loc='upper right')
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

def plot_reliability_correlations(df_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey=True)
    
    axes[0, 0].plot(df_results['SNR'], df_results['Rel_SA'], marker='x', color='black', linestyle='None')
    axes[0, 0].set_ylabel("SA reliability [%]")
    axes[0, 0].set_xlim([-20, 0])
    
    axes[0, 1].plot(df_results['SIR'], df_results['Rel_SA'], marker='x', color='black', linestyle='None')
    axes[0, 1].set_ylabel("SA reliability [%]")
    axes[0, 1].set_xlim([-35, -15])
    
    axes[1, 0].plot(df_results['SNR'], df_results['Rel_ICA'], marker='x', color='black', linestyle='None')
    axes[1, 0].set_xlabel("SNR [dB]")
    axes[1, 0].set_ylabel("ICA reliability [%]")
    
    axes[1, 1].plot(df_results['SIR'], df_results['Rel_ICA'], marker='x', color='black', linestyle='None')
    axes[1, 1].set_xlabel("SIR [dB]")
    axes[1, 1].set_ylabel("ICA reliability [%]")
    
    for ax in axes.flat:
        ax.set_ylim([-5, 105])
        
    fig.suptitle("FHR Detection Reliability vs SNR/SIR", fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()