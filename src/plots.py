import matplotlib.pyplot as plt
import numpy as np


channel_colors={"C3":"blue","C4":"red","Cz":"black"}
cond_colors={"T1":"green","T2":"purple","T0":"brown"}

def plot_raw(raw,t_start,t_end):
    freq=raw.info['sfreq']
    start=int(t_start*freq) #start index
    end=int(freq*t_end) #end index
    times=raw.times[start:end]
    raw_signal=raw.get_data()[0,start:end]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, raw_signal)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Raw Signal Segment: {t_start}s to {t_end}s')
    return fig

def plot_raw_vs_filtered( raw_filtered, pick_sensors, t_start, t_end):
    sfreq = raw_filtered.info['sfreq']
    start = int(t_start * sfreq)
    end   = int(t_end   * sfreq)
    times = raw_filtered.times[start:end]

    fig, axes = plt.subplots(figsize=(12, 5))
    for i, ch in enumerate(pick_sensors):
        filt_sig = raw_filtered.get_data(picks=[ch])[0, start:end] * 1e6
        axes.plot(times, filt_sig, color=channel_colors[ch], label=ch)
    axes.set_title("Filtered (8–30 Hz)")
    axes.set_ylabel("Amplitude (µV)")
    axes.set_xlabel("Time (s)")
    axes.legend()
    plt.tight_layout()
    return fig

def plot_psd(raw, pick_sensors):
    psd = raw.compute_psd(fmin=8, fmax=30, picks=pick_sensors, verbose=False)
    power, freqs = psd.get_data(return_freqs=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ch in enumerate(pick_sensors):
        ax.plot(freqs, 10 * np.log10(power[i]), color=channel_colors[ch], label=ch)
    ax.set_title("PSD — All Channels")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz)")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_psd_by_channel(psd_by_cond, pick_sensors):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, ch in zip(axes, pick_sensors):
        for cond in psd_by_cond:
            ch_names, avg_power, freqs = psd_by_cond[cond]
            i = ch_names.index(ch)
            ax.plot(freqs, 10 * np.log10(avg_power[i]),
                    color=cond_colors[cond], label=cond)
        ax.set_title(ch)
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()
    axes[0].set_ylabel("Power (dB/Hz)")
    fig.suptitle("PSD by Condition")
    plt.tight_layout()
    return fig
    