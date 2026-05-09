import mne
import os
from glob import glob
import numpy as np
from src.model_pipeline.preprocessing import renameChannels, bandFilter

def dataPrep(path_of_raw_datas):  #../data/raw/ in my case 
   
    # 1-20 subjects (4,8,12)- Runs each subject
    subjects=range(1,21)

    #X-> X-data (containing channels,signals)
    #Y-> Y-data (annotations (T1,T2))
    #groups -> subject matching each x,y rows 
    x_all,y_all,groups=[],[],[]

    event_id=dict(left=1,right=2)
    
    #channels 
    Channels=["C3","C4","Cz"]
    
    for s in subjects:
        #find all the eeg subject with .edf extension
        files=glob(os.path.join(path_of_raw_datas, f"S{s:03d}*.edf"))
        raw_file=[mne.io.read_raw_edf(i, preload=True)   for i in files]
        raw_concat=(mne.concatenate_raws(raw_file))
        #channels renamed C1. -> C1
        raw_concat=renameChannels(raw_concat)
        #filtered data from 8-30hz
        raw_filter=bandFilter(raw_concat)
        raw_filter.pick(Channels)
        
        #Annotation for T1,T2 -drop T0 
        event,event_id_full=mne.events_from_annotations(raw_filter,event_id=dict(T1=2,T2=3))
        #epoch for this subject with T1,T2 and 4 sec window
        epochs=mne.Epochs(
            raw_filter,
            event,
            event_id_full,
            tmin=0.8,
            tmax=3.5,
            baseline=None, 
            preload=True
        )
        x_all.append(epochs.get_data())
        y_all.append(epochs.events[:,-1]) #where t1,t2 is stored
        groups.append(np.full(len(epochs),s))

    X=np.concatenate(x_all)
    Y=np.concatenate(y_all)
    Groups=np.concatenate(groups)


    return X,Y,Groups

#raw_combined=dataPrep("../../data/raw/")
#duration_secs = raw_combined.n_times / raw_combined.info['sfreq']
#print(raw_combined.times[-1]) ##duration matches which proves concat works



