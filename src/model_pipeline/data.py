import mne
import os
from glob import glob
import numpy as np
from src.model_pipeline.preprocessing import renameChannels, bandFilter

def dataPrep(path_of_raw_datas=None, raw=None, from_file=True):

    Channels = ["C3", "C4", "Cz"]

    if from_file:
        subjects = [1, 2, 4, 7, 9, 13, 17, 29, 33, 34, 42, 47, 48, 49, 54, 55, 59, 60, 62, 69, 70, 72, 76, 83, 85]
        x_all, y_all, groups = [], [], []

        for s in subjects:
            files = glob(os.path.join(path_of_raw_datas, f"S{s:03d}*.edf"))
            raw_file = [mne.io.read_raw_edf(i, preload=True) for i in files]
            raw_concat = mne.concatenate_raws(raw_file)
            raw_concat = renameChannels(raw_concat)
            if raw_concat.info['sfreq'] != 160:
                raw_concat.resample(160, verbose=False)
            raw_filter = bandFilter(raw_concat)
            raw_filter.pick(Channels)

            event, event_id_full = mne.events_from_annotations(raw_filter, event_id=dict(T1=2, T2=3))
            epochs = mne.Epochs(
                raw_filter, event, event_id_full,
                tmin=0, tmax=4, baseline=None, preload=True, verbose=False
            )
            x_all.append(epochs.get_data())
            y_all.append(epochs.events[:, -1])
            groups.append(np.full(len(epochs), s))

        X      = np.concatenate(x_all)
        Y      = np.concatenate(y_all)
        Groups = np.concatenate(groups)

    else:
        raw_f = raw.copy()
        raw_f = renameChannels(raw_f)
        if raw_f.info['sfreq'] != 160:
            raw_f.resample(160, verbose=False)
        raw_f = bandFilter(raw_f)
        raw_f.pick(Channels)

        event, event_id_full = mne.events_from_annotations(raw_f, event_id=dict(T1=2, T2=3), verbose=False)
        epochs = mne.Epochs(
            raw_f, event, event_id_full,
            tmin=0, tmax=4, baseline=None, preload=True, verbose=False
        )

        X      = epochs.get_data()
        Y      = epochs.events[:, -1]
        Groups = np.zeros(len(epochs), dtype=int)

    return X, Y, Groups



