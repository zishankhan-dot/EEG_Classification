import matplotlib.pyplot as plt
import numpy as np
import mne 
import matplotlib.patches as mpatches


channel_colors={"C3":"blue","C4":"red","Cz":"black"}
cond_colors={"T1":"red","T2":"green","T0":"black"}


#plotting raw_file with annotations(t1,t2,t0) by default 10sec(0-10)
def plot_raw(raw, t_start=0, t_end=10, picks='C3'):
    s_freq  = raw.info['sfreq']
    s_index = int(t_start * s_freq)
    e_index = int(t_end * s_freq)
    signal  = raw.get_data(picks=picks)[0, s_index:e_index]
    times   = raw.times[s_index:e_index]

    fig, ax = plt.subplots(figsize=(5, 5.47))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(times, signal * 1e6, label=picks, color='#4C9BE8')

    event, event_id = mne.events_from_annotations(raw, verbose=False)
    label_id = {id: e for e, id in event_id.items()}
    for e in event:
        event_time = e[0] / s_freq
        if t_start < event_time < t_end:
            label = label_id[e[2]]
            ax.axvline(event_time, color=cond_colors.get(label, 'gray'), linestyle='-', linewidth=2)

    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in cond_colors.items()]
    ax.legend(handles=legend_handles, title="Event", loc="upper left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    return fig


def plotRawFilter(raw, t_start, t_end, picks="C3"):
    s_freq  = raw.info["sfreq"]
    s_index = int(t_start * s_freq)
    e_index = int(t_end * s_freq)
    times   = raw.times[s_index:e_index]
    signal  = raw.get_data(picks=picks)[0, s_index:e_index]

    filter_data   = raw.copy().filter(8.0, 30.0, verbose=False)
    signal_filter = filter_data.get_data(picks=picks)[0, s_index:e_index]

    fig, ax = plt.subplots(figsize=(5, 5.47))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(times, signal * 1e6, label="Raw", color='#4C9BE8')
    ax.plot(times, signal_filter * 1e6, label="Filtered (8–30 Hz)", color='orange', alpha=0.85)

    ax.legend(loc="upper left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    return fig



def plotMotorSensors(raw,picks=["C3","C4","Cz"]):
    raw=raw.copy()
    newNames={ch:ch.replace(".","") for ch in raw.ch_names}
    raw.rename_channels(newNames)
    raw.set_montage('standard_1005', match_case=False, on_missing='warn')

    fig=raw.plot_sensors(show_names=True, show=False)
    fig.set_size_inches(5, 5)
    ax=fig.axes[0]

    #overlay 
    position=raw.get_montage().get_positions()['ch_pos']
    for ch in picks:
        if ch in position:
            x,y,_=position[ch]
            ax.scatter(x,y,c='red',linewidth=2)
            ax.text(x,y,ch)

    ax.set_title("Motor Imagery Channels [C3,C4,Cz]")

    return fig


def plotConfusionMatrix(cm, labels=["Left (T1)", "Right (T2)"]):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=13)

    fig.tight_layout()
    return fig


def plotPredictionTimeline(table):
    color_map = {"Left (T1)": "#4C9BE8", "Right (T2)": "#F28C38"}
    n = len(table)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
    fig.patch.set_facecolor('white')

    for ax, col, title in [(ax1, 'Ground Truth', 'Ground Truth'), (ax2, 'Prediction', 'Prediction')]:
        ax.set_facecolor('white')
        for i, val in enumerate(table[col]):
            ax.barh(0, 1, left=i, color=color_map.get(val, 'gray'), edgecolor='white', linewidth=0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([title], fontsize=9)
        ax.set_xlim(0, n)
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')

    ax2.set_xlabel("Epoch", fontsize=9)

    handles = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    fig.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)
    fig.tight_layout()
    return fig