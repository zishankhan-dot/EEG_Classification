import mne
import matplotlib.pyplot as plt
import numpy as np


raw = mne.io.read_raw_edf("../data/raw/S001R04.edf", preload=True)

# snapping where channels are located on the scalp to 10*10 
# data is on 1010 but we mne doesnt have 1010 so i am gonna use 1005 which is superset of 1010 and has the same channels as 1010
# channel names in raw instance is different from the channel names in 1005 montage, changing the channel names in raw instance to match montage
newNames={ch:ch.replace(".","") for ch in raw.ch_names }
#print(newNames)
raw.rename_channels(newNames)
print(raw.ch_names)
# plotting the channels on the scalp 
#raw.set_montage('standard_1005', match_case=False, on_missing='warn')
#raw.plot_sensors(show_names=True)
#plt.show()


#using psd to visualize the data for different frequency bands
raw_psd=raw.compute_psd(fmin=8, fmax=30,picks=['C3','C4','Cz']) # max frequency is 30 because we are interested in alpha and beta bands which are between 8-30 Hz
plt.show()
channels=raw_psd.ch_names
power,freqs=raw_psd.get_data(return_freqs=True)


#plot it using matplotlib for each channel with different colors 
colors={'C3':'r','C4':'g','Cz':'b'}
for i,channel in enumerate(channels):
    plt.plot(freqs,10 * np.log10(power[i]),color=colors[channel],label=channel)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)') 
plt.title('Power Spectral Density for C3, C4 and Cz channels')
plt.legend()
plt.show()