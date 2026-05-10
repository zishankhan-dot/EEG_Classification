import mne
import os
from glob import glob
import numpy as np
from src.model_pipeline.preprocessing import renameChannels, bandFilter


def dataPrep2(path_of_raw_datas):
    subjects =range(1,21)

    x_all, y_all, groups = [], [], []

    Channels = ["C3", "C4", "Cz"]

    for s in subjects:
        for run in [4, 8, 12]:
            file = glob(os.path.join(path_of_raw_datas, f"S{s:03d}R{run:02d}.edf"))
            if not file:
                print(f"Subject {s:03d} Run {run:02d} | not found, skipping")
                continue

            raw = mne.io.read_raw_edf(file[0], preload=True, verbose=False)
            raw = renameChannels(raw)

            if raw.info['sfreq'] != 160:
                raw.resample(160, verbose=False)

            raw_filter = bandFilter(raw)
            raw_filter.pick(Channels)

            event, event_id_full = mne.events_from_annotations(
                raw_filter, event_id=dict(T1=2, T2=3), verbose=False
            )

            epochs = mne.Epochs(
                raw_filter,
                event,
                event_id_full,
                tmin=2.0,
                tmax=6.0,
                baseline=None,
                preload=True,
                reject=None,
                verbose=False
            )

            if len(epochs) == 0:
                print(f"Subject {s:03d} Run {run:02d} | 0 epochs, skipping")
                continue

            x_all.append(epochs.get_data())
            y_all.append(epochs.events[:, -1])
            groups.append(np.full(len(epochs), s))
            print(f"Subject {s:03d} Run {run:02d} | {len(epochs)} epochs")

    X = np.concatenate(x_all)
    Y = np.concatenate(y_all)
    Groups = np.concatenate(groups)

    return X, Y, Groups