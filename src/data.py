import mne 


raw = mne.io.read_raw_edf("../data/raw/S001R04.edf", preload=True)
print(raw.info)              # sampling freq, channels, etc.
print(raw.ch_names)          # list of channel names
print(raw.times[-1])         # recording duration in seconds
print(raw.annotations)

