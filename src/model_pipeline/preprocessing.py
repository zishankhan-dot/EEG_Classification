import numpy as np
from scipy.signal import welch

def renameChannels(raw_combined):
    #changing ch_names from 1010 standard  "C1." -> "C1" 
    mapping={ch:ch.replace(".","") for ch in raw_combined.ch_names }
    raw_combined.rename_channels(mapping)
    
    return raw_combined

def bandFilter(raw_combined):
    raw_filter = raw_combined.filter(8., 30., fir_design='firwin', verbose=False)
    return raw_filter


def featureExtraction(X,relative=False,s_freq=160,bands={'mu':(8,13),'beta':(13,30)}):
    n_trails,channels,_=X.shape
    features=[] #3 channels * 2 target(t1,t2) => 6 bands 2(alpha and beta) for each channels

    #welch 
    freq, psd=welch(X,fs=s_freq,axis=-1)
    print(psd.shape)

    #total_power for relative calc
    total_power=psd.mean(axis=-1)
    
    #band-based mu & beta
    for band,(min,max) in bands.items():
        mask=(freq>=min)&(freq<=max)
        bandpower=psd[:,:,mask].mean(axis=-1) # change freqs_bins -> 1 per n_samples
        #relative true
        if relative:
            bandpower=bandpower/total_power

        features.append(np.log(bandpower)) #to make normalize (reduce uncertain peaks) # (n_samples,n_channels * n_bands)
    
    #final feature array
    return np.concatenate(features,axis=1)



